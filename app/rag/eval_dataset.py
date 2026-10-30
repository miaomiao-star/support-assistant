"""读取用户标注并计算分组指标；仅供离线评估，不参与线上检索或生成。"""

import json
import re
import unicodedata
from pathlib import Path
from app.rag.evaluation import mean_or_none
from app.rag.retrieval import STRATEGIES


def load_cases(path):
    """逐行读取JSONL，拒绝重复题号和缺失标注，原题目/标签保持不变。"""
    rows = []
    seen = set()
    for number, line in enumerate(
        Path(path).read_text(encoding='utf-8-sig').splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'第{number}行JSON无效') from exc
        for field in (
            'id',
            'bucket',
            'query',
            'expect_section',
            'expect_points',
            'should_refuse',
        ):
            if field not in row:
                raise ValueError(f'第{number}行缺少{field}')
        if not all(
            isinstance(row[k], str) and row[k].strip()
            for k in ('id', 'bucket', 'query')
        ):
            raise ValueError(f'第{number}行题号/分类/问题无效')
        if row['id'] in seen:
            raise ValueError(f'重复题号：{row["id"]}')
        if type(row['should_refuse']) is not bool:
            raise ValueError(f'第{number}行should_refuse须为布尔值')
        for key in ('expect_section', 'expect_points'):
            if not isinstance(row[key], list) or any(
                not isinstance(v, str) or not v.strip() for v in row[key]
            ):
                raise ValueError(f'第{number}行{key}须为非空文本列表')
        groups = row.get('expect_sections_all')
        if groups is not None and (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(g, list)
                or not g
                or any(not isinstance(v, str) or not v.strip() for v in g)
                for g in groups
            )
        ):
            raise ValueError(f'第{number}行expect_sections_all无效')
        if not row['should_refuse'] and not row['expect_section']:
            raise ValueError(f'第{number}行可答题没有标注来源')
        seen.add(row['id'])
        rows.append(row)
    if not rows:
        raise ValueError('评估集为空')
    return rows


def normalized_title(value):
    """只统一全半角、大小写和空格，不做会改变来源含义的语义猜测。"""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value)).casefold()


def label_ids(label, chunks, aliases):
    """用审阅后的标题别名定位原文；型号必须完整匹配，MH-W40不能匹配MH-W400。"""
    if re.fullmatch(r'MH-[A-Za-z0-9]+', label, re.I):
        pattern = re.compile(
            r'(?<![A-Za-z0-9-])' + re.escape(label) + r'(?![A-Za-z0-9-])', re.I
        )
        return {
            r['id']
            for r in chunks
            if pattern.search(r['questions'] + ' ' + (r.get('section_path') or ''))
        }
    titles = {normalized_title(v) for v in aliases.get(label, [label])}
    matched = set()
    for row in chunks:
        # 问法可能因挖知识追加多行；章节路径末级才是标题，不用上级路径作宽泛命中。
        choices = row['questions'].splitlines() + [
            (row.get('section_path') or '').rsplit(' / ', 1)[-1]
        ]
        if titles.intersection(normalized_title(v) for v in choices):
            matched.add(row['id'])
    return matched


def resolve_groups(case, chunks, aliases):
    """外层每一组都必须命中，内层任一来源可接受；缺失组保留空ids计作未命中。"""
    labels = case.get('expect_sections_all') or (
        [case['expect_section']] if case['expect_section'] else []
    )
    groups = []
    unmapped = []
    for alternatives in labels:
        ids = set()
        for label in alternatives:
            found = label_ids(label, chunks, aliases)
            if not found:
                unmapped.append(label)
            ids.update(found)
        groups.append({'labels': alternatives, 'ids': sorted(ids)})
    return {
        'groups': groups,
        'relevant_ids': sorted({key for group in groups for key in group['ids']}),
        'unmapped_labels': sorted(set(unmapped)),
        'fully_mapped': all(group['ids'] for group in groups),
    }


def group_metrics(rows, groups, ks=(1, 5, 10)):
    """Recall按证据组计算，避免把可替代的章节错误地当成必须全部召回的块。"""
    if not groups:
        return None
    ids = [r['id'] for r in rows]
    result = {
        f'recall@{k}': sum(bool(set(ids[:k]) & set(g['ids'])) for g in groups)
        / len(groups)
        for k in ks
    }
    relevant = {key for g in groups for key in g['ids']}
    result['mrr@10'] = next(
        (1 / (index + 1) for index, key in enumerate(ids[:10]) if key in relevant), 0.0
    )
    return result


def point_coverage(points, outcome):
    """辅助字面覆盖：只忽略空白/标点，不把它包装成语义正确率。"""

    def normalize(text):
        return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', text)).casefold()

    answer = normalize(outcome['answer']) if outcome['useful'] else ''
    details = []
    for point in points:
        normalized_point = normalize(point)
        details.append(
            {
                'point': point,
                'matched': bool(normalized_point) and normalized_point in answer,
            }
        )
    return {
        'coverage': (
            sum(p['matched'] for p in details) / len(details) if details else None
        ),
        'details': details,
    }


def summarize_dataset(rows):
    """按用户的可答/拒答标注分桶；调用失败单列，不能因来源缺失而删掉正例。"""
    result = {}
    for strategy in STRATEGIES:
        group = [r for r in rows if r['strategy'] == strategy]
        result[strategy] = {}
        for bucket in ['all'] + sorted({r['bucket'] for r in group}):
            selected = [r for r in group if bucket == 'all' or r['bucket'] == bucket]
            completed = [r for r in selected if 'error' not in r]
            answerable = [r for r in completed if not r['should_refuse']]
            unanswerable = [r for r in completed if r['should_refuse']]
            judged = [
                r['faith']['faithfulness']
                for r in completed
                if r['faith']['faithfulness'] is not None
            ]
            metrics = [r['metrics'] for r in answerable if r['metrics'] is not None]
            covered = [r for r in answerable if r['mapping']['fully_mapped']]
            points = [
                r['points']['coverage']
                for r in answerable
                if r['points']['coverage'] is not None
            ]

            result[strategy][bucket] = {
                'total': len(selected),
                'completed': len(completed),
                'errors': len(selected) - len(completed),
                **(
                    {key: mean_or_none([m[key] for m in metrics]) for key in metrics[0]}
                    if metrics
                    else {}
                ),
                'faithfulness': mean_or_none(judged),
                'faithfulness_answered_n': len(judged),
                'answerable_count': len(answerable),
                'answerable_answered': sum(r['outcome']['useful'] for r in answerable),
                'false_refusals': sum(not r['outcome']['useful'] for r in answerable),
                'refusal_count': len(unanswerable),
                'refusal_correct': sum(
                    not r['outcome']['useful'] for r in unanswerable
                ),
                'failed_refusals': sum(r['outcome']['useful'] for r in unanswerable),
                'point_literal_coverage': mean_or_none(points),
                'fully_mapped_cases': len(covered),
                'mapped_recall@10': mean_or_none(
                    [r['metrics']['recall@10'] for r in covered]
                ),
                'mapped_mrr@10': mean_or_none(
                    [r['metrics']['mrr@10'] for r in covered]
                ),
            }
    return result
