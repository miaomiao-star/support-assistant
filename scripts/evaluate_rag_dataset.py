"""原样评估用户JSONL：四策略、组合证据、拒答、忠实度和逐点字面覆盖。"""

import argparse
import ast
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.rag.eval_dataset import (
    load_cases,
    resolve_groups,
    group_metrics,
    point_coverage,
    summarize_dataset,
)
from app.rag.retrieval import STRATEGIES


def digest(value):
    """稳定摘要用于校验数据/原文版本，不把数据库地址或密钥写进报告。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def ledger_id(case_id):
    """300题台账加固定前缀，避免A10等题号覆盖之前50题的历史个案。"""
    value = 'e300-' + case_id
    if len(value) > 16 or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('评估题号超出台账容量或格式无效')
    return value


def load_checkpoint(path, version, allow_execution_change=False):
    """日志逐条落盘，因此即使汇总快照较旧，也能找回已完成的最后几条结果。"""
    path = Path(path)
    source = json.loads(path.read_text(encoding='utf-8'))
    before = json.loads(json.dumps(source['version']))
    after = json.loads(json.dumps(version))
    if allow_execution_change:
        # 只允许显式忽略运行器自身的变化；模型、Prompt、计分、原文仍须严格一致。
        for value in (before, after):
            value.get('core_ast', {}).pop('scripts/evaluate_rag_dataset.py', None)
    if before != after:
        raise ValueError('数据、别名、原文或核心代码版本变化，不能混合续跑')
    journal = path.with_suffix('.jsonl')
    rows = (
        [
            json.loads(line)
            for line in journal.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        if journal.exists()
        else source['results']
    )
    keys = [(row['id'], row['strategy']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('断点日志含重复题目/策略')
    return rows


async def rate_limited_call(operation, stage, errors, delays=(10, 20, 40)):
    """仅对429/限流做有上限退避；不得因为答案拒答或裁判低分就重新生成。"""
    for attempt in range(len(delays) + 1):
        try:
            return await operation()
        except Exception as exc:
            limited = (
                getattr(exc, 'status_code', None) == 429
                or 'ratelimit' in type(exc).__name__.lower()
            )
            if not limited:
                raise
            retry = attempt < len(delays)
            errors.append(
                {
                    'stage': stage,
                    'attempt': attempt + 1,
                    'type': type(exc).__name__,
                    'retry': retry,
                }
            )
            if not retry:
                raise
            await asyncio.sleep(delays[attempt])


def save_report(path, report):
    """先写同目录临时文件再替换，防止中断留下半截JSON；逐条日志仍独立保留。"""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8'
    )
    temporary.replace(path)


def markdown_report(path, report):
    """报告同时展示来源缺口与已覆盖来源子集，不能只保留较好看的数字。"""

    def fmt(v):
        return 'N/A' if v is None else f'{v:.3f}'

    lines = [
        '# Ch04 用户300题真实评估',
        f'原始集：{report["dataset"]}',
        f'逐条结果：{path.with_suffix(".jsonl").name}',
        f'成功{report["successful_results"]}/{report["expected_results"]}，错误{report["error_results"]}。',
        '',
        '四策略共享Query理解；线上知识/Prompt/模型/门槛不因本集调参。',
        'Recall按证据组计分：外层AND、组内OR，缺失来源组计未命中。',
        'Faithfulness沿用当前聊天模型自动裁判，拒答N/A；字面覆盖只忽略空白/标点，不是语义准确率。',
        '',
        '|策略|桶|完成/总数|Recall@10|MRR@10|Faithfulness|可答题作答|应拒答正确|字面覆盖|',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for strategy, buckets in report['summary'].items():
        for bucket, m in buckets.items():
            lines.append(
                '|'
                + '|'.join(
                    [
                        strategy,
                        bucket,
                        f'{m["completed"]}/{m["total"]}',
                        fmt(m.get('recall@10')),
                        fmt(m.get('mrr@10')),
                        fmt(m['faithfulness']),
                        f'{m["answerable_answered"]}/{m["answerable_count"]}',
                        f'{m["refusal_correct"]}/{m["refusal_count"]}',
                        fmt(m['point_literal_coverage']),
                    ]
                )
                + '|'
            )
    audit = report['source_audit']
    lines += [
        '',
        '## 来源覆盖与标注差异',
        f'标注可答题{audit["answerable"]}，其中来源组全部可定位{audit["fully_mapped"]}。',
        '当前未找到的章节标签：' + ('、'.join(audit['unmapped_labels']) or '无'),
        '标题别名只用于评估来源定位，不给检索器或生成器读取。缺少具体细则仍可能导致拒答。',
        '',
        '|策略|来源完全覆盖题数|条件Recall@10|条件MRR@10|',
        '|---|---|---|---|',
    ]
    for strategy, buckets in report['summary'].items():
        m = buckets['all']
        lines.append(
            f'|{strategy}|{m["fully_mapped_cases"]}|{fmt(m["mapped_recall@10"])}|{fmt(m["mapped_mrr@10"])}|'
        )
    lines += [
        '',
        '## 正式策略逐题问题清单',
        '“漏拒答”按用户should_refuse与线上useful信号计分；解释性否定也可能被模型标为useful，须看原答案。',
    ]
    for row in report['results']:
        if row['strategy'] != 'hybrid_rerank':
            continue
        reasons = []
        if 'error' in row:
            reasons.append('调用错误：' + row['error'])
        else:
            if row['should_refuse'] == row['outcome']['useful']:
                reasons.append(
                    '漏拒答' if row['should_refuse'] else '误拒答（按原标注）'
                )
            if not row['should_refuse'] and row['metrics']['recall@10'] < 1:
                reasons.append('证据组未全部召回')
            if row['faith']['unsupported']:
                reasons.append('有不支持事实')
            if row['mapping']['unmapped_labels']:
                reasons.append(
                    '标注来源缺口：' + ','.join(row['mapping']['unmapped_labels'])
                )
        if reasons:
            lines.append(
                f'- {row["id"]}：'
                + '；'.join(reasons)
                + '。问题：'
                + row['query'].replace('\n', ' ')
            )
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


async def main(args):
    """先做无付费调用的来源预检，再对四策略运行真实检索/生成/忠实度裁判。"""
    # 延迟导入服务，纯计分和断点单测不需要读取应用配置或创建客户端。
    from app.db import Database, DatabaseSettings
    from app.knowledge.runtime import build_service
    from app.config import get_settings
    from app.core.llm import get_chat_model
    from app.rag.quality import judge_answer

    if args.concurrency < 1 or args.case_workers < 1:
        raise ValueError('并发必须是正数')
    dataset = Path(args.dataset).resolve()
    alias_path = ROOT / 'tests/data/ch04_section_aliases.json'
    cases = load_cases(dataset)
    aliases = json.loads(alias_path.read_text(encoding='utf-8'))
    for case in cases:
        ledger_id(case['id'])
    settings = get_settings()
    db = Database(DatabaseSettings().url())
    service = build_service(db, True)
    path = ROOT / 'dev-notes' / f'ch04-eval300-{time.time_ns()}.json'
    try:
        await db.check()
        chunks = await service.store.all_chunks()
        mappings = {case['id']: resolve_groups(case, chunks, aliases) for case in cases}
        positive = [c for c in cases if not c['should_refuse']]
        audit = {
            'total': len(cases),
            'buckets': dict(Counter(c['bucket'] for c in cases)),
            'answerable': len(positive),
            'refusal': len(cases) - len(positive),
            'fully_mapped': sum(mappings[c['id']]['fully_mapped'] for c in positive),
            'unmapped_labels': sorted(
                {v for mapping in mappings.values() for v in mapping['unmapped_labels']}
            ),
            'mappings': mappings,
        }
        # AST摘要不受普通注释影响；实际Prompt字符串和代码行为仍包含在摘要中。
        core_names = [
            'app/rag/retrieval.py',
            'app/rag/quality.py',
            'app/rag/vector.py',
            'app/rag/eval_dataset.py',
            'scripts/evaluate_rag_dataset.py',
        ]
        core = {
            name: digest(
                ast.dump(
                    ast.parse((ROOT / name).read_text(encoding='utf-8')),
                    include_attributes=False,
                )
            )
            for name in core_names
        }
        version = {
            'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
            'aliases_sha256': digest(aliases),
            'corpus_sha256': digest(chunks),
            'core_ast': core,
            'min_rerank_score': settings.rag_min_score,
            'chat_model': settings.chat_model,
            'embed_model': settings.embed_model,
            'rerank_model': settings.rerank_model or 'BAAI/bge-reranker-v2-m3',
        }
        report = {
            'dataset': str(dataset),
            'version': version,
            'source_audit': audit,
            'corpus_snapshot': chunks,
            'aliases': aliases,
            'results': [],
            'expected_results': len(cases) * len(STRATEGIES),
            'judge_limit': '同一聊天模型自动裁判，不是独立人工准确率；拒答Faithfulness为N/A',
            'execution': {
                'concurrency': args.concurrency,
                'case_workers': args.case_workers,
            },
            'run_complete': False,
        }
        if args.preflight:
            path = path.with_name(path.stem + '-preflight.json')
            save_report(path, report)
            print(
                json.dumps(
                    {k: v for k, v in audit.items() if k != 'mappings'},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(path, flush=True)
            return
        prior = (
            load_checkpoint(args.resume, version, args.resume_execution_change)
            if args.resume
            else []
        )
        rows = [r for r in prior if not (args.retry_errors and 'error' in r)]
        allowed = {(case['id'], s) for case in cases for s in STRATEGIES}
        if any((r['id'], r['strategy']) not in allowed for r in rows):
            raise ValueError('断点结果含本集以外的题目或策略')
        done = {(r['id'], r['strategy']) for r in rows}
        # 从日志恢复完全相同的Query理解，四策略不得在续跑时各自重新改写。
        cache = service.retriever.understanding.cache
        for row in prior:
            if row.get('understanding'):
                if (
                    row['query'] in cache
                    and cache[row['query']] != row['understanding']
                ):
                    raise ValueError('同题Query理解不一致')
                cache[row['query']] = row['understanding']
        report['results'] = rows
        if args.resume:
            report['resumed_from'] = str(Path(args.resume).resolve())
        if args.retry_errors:
            # 旧错误留在原报告和新报告元数据中，重跑后也不能伪装为从未发生。
            report['previous_errors'] = [
                {'id': r['id'], 'strategy': r['strategy'], 'error': r['error']}
                for r in prior
                if 'error' in r
            ]
        report['execution']['rate_limit_backoff_seconds'] = [10, 20, 40]
        report['execution']['resume_execution_change'] = args.resume_execution_change
        journal = path.with_suffix('.jsonl')
        journal.write_text(
            ''.join(
                json.dumps(row, ensure_ascii=False, default=str) + '\n' for row in rows
            ),
            encoding='utf-8',
        )
        save_report(path, report)
        print('REPORT ' + str(path), flush=True)
        semaphore = asyncio.Semaphore(args.concurrency)
        query_locks = {case['query']: asyncio.Lock() for case in cases}
        start = time.monotonic()
        model = get_chat_model()

        def record(row):
            # 先逐条关闭日志文件完成落盘，再定期保存汇总，不丢最后一个已完成结果。
            rows.append(row)
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
            if len(rows) % 20 == 0 or 'error' in row:
                report['summary'] = summarize_dataset(rows)
                save_report(path, report)
                print(f'PROGRESS {len(rows)}/{report["expected_results"]}', flush=True)

        async def evaluate(case, strategy):
            async with semaphore:
                row = {
                    **case,
                    'strategy': strategy,
                    'mapping': mappings[case['id']],
                    'understanding': dict(cache[case['query']]),
                    'operation_errors': [],
                }
                began = time.monotonic()
                try:
                    retrieval = await rate_limited_call(
                        lambda: service.retriever.search(
                            case['query'], case.get('filters'), strategy
                        ),
                        'retrieval',
                        row['operation_errors'],
                    )
                    row['metrics'] = group_metrics(
                        retrieval['ranked'], row['mapping']['groups']
                    )
                    if row['metrics'] is not None:
                        row['metrics']['candidate_recall@50'] = group_metrics(
                            retrieval['candidates'][:50],
                            row['mapping']['groups'],
                            (50,),
                        )['recall@50']
                    # 只把模型实际看到的原文放在outcome.citations；候选用ID+分数引用共享原文快照。
                    row['retrieval'] = {
                        'low_confidence': retrieval['low_confidence'],
                        'candidates': [
                            {'id': r['id'], 'score': r['score']}
                            for r in retrieval['candidates']
                        ],
                        'ranked': [
                            {k: r[k] for k in ('id', 'score', 'rerank_score') if k in r}
                            for r in retrieval['ranked']
                        ],
                    }
                    row['outcome'] = await rate_limited_call(
                        lambda: service.generator.generate(case['query'], retrieval),
                        'generate',
                        row['operation_errors'],
                    )
                    row['faith'] = await rate_limited_call(
                        lambda: judge_answer(model, case['query'], row['outcome']),
                        'judge',
                        row['operation_errors'],
                    )
                    row['points'] = point_coverage(
                        case['expect_points'], row['outcome']
                    )
                    if (
                        strategy == 'hybrid_rerank'
                        and row['faith']['faithfulness'] is not None
                        and row['faith']['faithfulness'] < 1
                    ):
                        await service.quality.record_faith_case(
                            {
                                'eval_id': ledger_id(case['id']),
                                'bucket': case['bucket'],
                                'query': case['query'],
                                'strategy': strategy,
                                'answer': row['outcome']['answer'],
                                'reason': (
                                    json.dumps(
                                        row['faith']['unsupported'], ensure_ascii=False
                                    )
                                    if row['faith']['claims']
                                    else '裁判无有效事实，需人工复核'
                                ),
                                'citations': [
                                    {
                                        k: c[k]
                                        for k in (
                                            'n',
                                            'chunk_id',
                                            'section_path',
                                            'question',
                                            'answer',
                                        )
                                    }
                                    for c in row['outcome']['citations']
                                ],
                                'judge_model': settings.chat_model,
                            }
                        )
                except Exception as exc:
                    row['error'] = type(exc).__name__
                row['elapsed_seconds'] = round(time.monotonic() - began, 3)
                record(row)

        async def evaluate_case(case):
            missing = [s for s in STRATEGIES if (case['id'], s) not in done]
            if not missing:
                return
            query_errors = []
            try:
                # 相同原话出现在不同桶时也只做一次理解，避免并发首次请求覆盖缓存。
                async with query_locks[case['query']]:
                    await rate_limited_call(
                        lambda: service.retriever.understanding.understand(
                            case['query']
                        ),
                        'understanding',
                        query_errors,
                    )
            except Exception as exc:
                for strategy in missing:
                    record(
                        {
                            **case,
                            'strategy': strategy,
                            'mapping': mappings[case['id']],
                            'error': 'QueryUnderstanding:' + type(exc).__name__,
                            'operation_errors': query_errors,
                        }
                    )
                return
            await asyncio.gather(*(evaluate(case, s) for s in missing))

        for offset in range(0, len(cases), args.case_workers):
            await asyncio.gather(
                *(evaluate_case(c) for c in cases[offset : offset + args.case_workers])
            )
        report['summary'] = summarize_dataset(rows)
        report['error_results'] = sum('error' in row for row in rows)
        report['successful_results'] = len(rows) - report['error_results']
        report['run_complete'] = len(rows) == report['expected_results']
        report['corpus_unchanged'] = (
            digest(await service.store.all_chunks()) == version['corpus_sha256']
        )
        report['elapsed_seconds'] = round(time.monotonic() - start, 3)
        save_report(path, report)
        markdown_report(path, report)
        print('FINISHED ' + str(path), flush=True)
    finally:
        await service.close()
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=str(ROOT / 'tests/data/eval_ch04.jsonl'))
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--case-workers', type=int, default=4)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', default='')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--resume-execution-change', action='store_true')
    asyncio.run(main(parser.parse_args()))
