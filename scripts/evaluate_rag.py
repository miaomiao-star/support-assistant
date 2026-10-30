"""同一标注集与Query理解比较四策略，逐题留报告；评估不污染业务拒答池。"""

import argparse
import asyncio
import json
import sys
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.db import Database, DatabaseSettings
from app.knowledge.runtime import build_service
from app.config import get_settings
from app.core.llm import get_chat_model
from app.rag.retrieval import STRATEGIES
from app.rag.evaluation import mean_or_none, retrieval_metrics
from app.rag.quality import judge_answer


def resume_state(source, dataset_hash, code_hashes):
    """断点续跑只复用同一数据/核心实现的结果，并恢复每题同一份Query理解。"""
    if source.get('dataset_sha256') != dataset_hash:
        raise ValueError('标注版本已变化，不能混入旧评估')
    for name, digest in code_hashes.items():
        if (
            name.startswith('app/rag/')
            and source.get('code_sha256', {}).get(name) != digest
        ):
            raise ValueError('检索或生成实现变化，必须重新完整评估')
    done = set()
    cache = {}
    for row in source['results']:
        key = (row['id'], row['strategy'])
        if key in done:
            raise ValueError('断点评估有重复题目/策略')
        done.add(key)
        if row.get('retrieval'):
            query = row['retrieval']['query']
            if row['query'] in cache and cache[row['query']] != query:
                raise ValueError('同题Query理解不一致')
            cache[row['query']] = query
    return {'done': done, 'cache': cache}


def summarize(rows):
    """指标仅对成功题计分，同时明确错误数和覆盖率，避免漏跑题被悄悄丢弃。"""
    result = {}
    for strategy in STRATEGIES:
        result[strategy] = {}
        group = [r for r in rows if r['strategy'] == strategy]
        for bucket in ['all'] + sorted({r['bucket'] for r in group}):
            selected = [r for r in group if bucket == 'all' or r['bucket'] == bucket]
            completed = [r for r in selected if 'error' not in r]
            metrics = [r['metrics'] for r in completed if r['metrics']]
            # Faithfulness只平均实际作答的题；答案覆盖和拒答率必须另外看。
            faith = [
                r['faith']['faithfulness']
                for r in completed
                if r['faith']['faithfulness'] is not None
            ]
            unanswerable = [r for r in completed if not r['relevant_ids']]
            answerable = [r for r in completed if r['relevant_ids']]
            result[strategy][bucket] = {
                'total': len(selected),
                'completed': len(completed),
                'errors': len(selected) - len(completed),
                **(
                    {key: mean_or_none([m[key] for m in metrics]) for key in metrics[0]}
                    if metrics
                    else {}
                ),
                'faithfulness': mean_or_none(faith),
                'faithfulness_answered_n': len(faith),
                'answer_rate': mean_or_none(
                    [r['outcome']['useful'] for r in completed]
                ),
                'unknown_refusal_rate': mean_or_none(
                    [not r['outcome']['useful'] for r in unanswerable]
                ),
                'answerable_answer_rate': mean_or_none(
                    [r['outcome']['useful'] for r in answerable]
                ),
            }
    return result


async def main(args):
    if args.concurrency < 1 or args.case_workers < 1 or args.limit < 0:
        raise ValueError('并发必须为正数，limit不能为负数')
    db = Database(DatabaseSettings().url())
    service = build_service(db, rag_enabled=True)
    model = get_chat_model()
    settings = get_settings()
    stamp = time.time_ns()
    path = ROOT / 'dev-notes' / f'ch04-evaluation-{stamp}.json'
    report = {
        'dataset': 'tests/fixtures/ch04_eval.json',
        'judge_model': settings.chat_model,
        'judge_limit': '同一上游模型自评，非独立人工裁判；拒答Faithfulness为N/A',
        'min_rerank_score': settings.rag_min_score,
        'results': [],
    }
    # 固定记录本次标注与实现摘要，后续修订不会使旧报告看起来来自最新版本。
    report['dataset_sha256'] = hashlib.sha256(
        (ROOT / report['dataset']).read_bytes()
    ).hexdigest()
    report['code_sha256'] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            'app/rag/retrieval.py',
            'app/rag/quality.py',
            'app/rag/vector.py',
            'scripts/evaluate_rag.py',
        )
    }
    try:
        cases = json.loads((ROOT / report['dataset']).read_text(encoding='utf-8'))
        # 本入口保留最初50题JSON约定；用户300题JSONL使用evaluate_rag_dataset.py。
        if args.limit:
            cases = cases[: args.limit]
        chunks = await service.store.all_chunks()
        for case in cases:
            # 标注章节名在入库时解析为真实稳定主键；标注找不到时立即失败而非空真值。
            ids = []
            for label in case['evidence']:
                matched = [
                    r['id']
                    for r in chunks
                    if (r['section_path'] or '').endswith(' / ' + label)
                    or label in r['questions']
                    and label.startswith('型号 ')
                ]
                if not matched:
                    raise ValueError('标注证据未找到：' + case['id'] + '/' + label)
                ids.extend(matched)
            case['relevant_ids'] = sorted(set(ids))
        report['corpus'] = [
            {'id': r['id'], 'section_path': r['section_path']} for r in chunks
        ]
        done = set()
        if args.resume:
            source = json.loads(Path(args.resume).read_text(encoding='utf-8'))
            state = resume_state(
                source, report['dataset_sha256'], report['code_sha256']
            )
            # 旧报告已有的正文快照必须与当前权威源一致；禁止后台改库后混跑。
            known = {r['id']: r for r in chunks}
            prior = {
                r['id']: r
                for item in source['results']
                for r in item.get('retrieval', {}).get('candidates', [])
            }
            if set(prior) != set(known) or any(
                any(
                    prior[key].get(f) != known[key].get(f)
                    for f in (
                        'category',
                        'questions',
                        'answer',
                        'section_path',
                        'content_type',
                    )
                )
                for key in known
            ):
                raise ValueError('原文快照已变化或覆盖不全，不能续跑')
            report['results'] = source['results']
            done = state['done']
            # 续跑不会重新理解已完成过的题，保证同一题四策略的检索输入相同。
            service.retriever.understanding.cache.update(state['cache'])
            report['resumed_from'] = str(args.resume)
        report['execution'] = {
            'strategy_concurrency': args.concurrency,
            'case_workers': args.case_workers,
        }
        # 每题先顺序完成理解，再并发四策略，防止缓存首次并发造成理解四次。
        semaphore = asyncio.Semaphore(args.concurrency)

        async def evaluate(case, strategy):
            async with semaphore:
                row = {**case, 'strategy': strategy}
                try:
                    retrieval = await service.retriever.search(
                        case['query'], case.get('filters'), strategy
                    )
                    row['metrics'] = retrieval_metrics(
                        retrieval['ranked'], case['relevant_ids']
                    )
                    if row['metrics'] is not None:
                        row['metrics']['candidate_recall@50'] = retrieval_metrics(
                            retrieval['candidates'][:50], case['relevant_ids'], (50,)
                        )['recall@50']
                    row['retrieval'] = retrieval
                    row['outcome'] = await service.generator.generate(
                        case['query'], retrieval
                    )
                    # 评估直接调用生成服务，不经过ChatService，因此不写真实聊天拒答池。
                    row['faith'] = await judge_answer(
                        model, case['query'], row['outcome']
                    )
                    if (
                        strategy == 'hybrid_rerank'
                        and row['faith']['faithfulness'] is not None
                        and row['faith']['faithfulness'] < 1
                    ):
                        # 只有正式策略写长期台账，其他三路留在该次报告中，避免相互覆盖。
                        await service.quality.record_faith_case(
                            {
                                'eval_id': case['id'],
                                'bucket': case['bucket'],
                                'query': case['query'],
                                'strategy': strategy,
                                'answer': row['outcome']['answer'],
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
                                'reason': (
                                    json.dumps(
                                        row['faith']['unsupported'], ensure_ascii=False
                                    )
                                    if row['faith']['claims']
                                    else '裁判未返回原子事实，需人工复核'
                                ),
                                'judge_model': settings.chat_model,
                            }
                        )
                except Exception as exc:
                    row['error'] = type(exc).__name__
                # 一题调用失败也记入报告；不能只输出成功题造成虚假的高覆盖。
                report['results'].append(row)
                report['summary'] = summarize(report['results'])
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8',
                )
                print(
                    case['id'],
                    strategy,
                    'ERROR ' + row['error'] if 'error' in row else 'ok',
                    flush=True,
                )

        async def evaluate_case(case):
            missing = [s for s in STRATEGIES if (case['id'], s) not in done]
            if not missing:
                return
            try:
                await service.retriever.understanding.understand(case['query'])
            except Exception as exc:
                for strategy in missing:
                    report['results'].append(
                        {
                            **case,
                            'strategy': strategy,
                            'error': 'QueryUnderstanding:' + type(exc).__name__,
                        }
                    )
                report['summary'] = summarize(report['results'])
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding='utf-8',
                )
                return
            await asyncio.gather(*(evaluate(case, s) for s in missing))

        # 小批并行题目消除“某一策略慢时另外三路空闲”，总管线并发仍受semaphore约束。
        for offset in range(0, len(cases), args.case_workers):
            await asyncio.gather(
                *(
                    evaluate_case(case)
                    for case in cases[offset : offset + args.case_workers]
                )
            )
        report['completed'] = len(report['results']) == len(cases) * 4 and all(
            'error' not in r for r in report['results']
        )
        # JSON保存逐题证据和原子判断，Markdown只提供可快速阅读的分桶摘要。
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        lines = [
            '# Ch04 四策略真实评估',
            f'报告：{path.name}',
            '',
            report['judge_limit'],
            '',
            '|策略|桶|完成/总数|Recall@10|MRR@10|Faithfulness|可答题作答率|无答案拒答率|',
            '|---|---|---|---|---|---|---|---|',
        ]

        def fmt(value):
            return 'N/A' if value is None else f'{value:.3f}'

        for strategy, groups in report['summary'].items():
            for bucket, m in groups.items():
                lines.append(
                    '|'
                    + '|'.join(
                        [
                            strategy,
                            bucket,
                            f"{m['completed']}/{m['total']}",
                            fmt(m.get('recall@10')),
                            fmt(m.get('mrr@10')),
                            fmt(m['faithfulness']),
                            fmt(m['answerable_answer_rate']),
                            fmt(m['unknown_refusal_rate']),
                        ]
                    )
                    + '|'
                )
        path.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(path, flush=True)
    finally:
        await service.close()
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=2)
    parser.add_argument('--case-workers', type=int, default=2)
    parser.add_argument('--resume', type=str, default='')
    asyncio.run(main(parser.parse_args()))
