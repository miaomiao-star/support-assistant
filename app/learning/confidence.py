"""可复现经验证据分：只用正式重排信号，阈值在冻结标注集训练分区选择。"""

from copy import deepcopy
import hashlib
import math


def scores(rows):
    values = []
    for row in rows:
        value = row.get('rerank_score')
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError('证据缺少有限的正式重排分数')
        values.append(float(value))
    if values != sorted(values, reverse=True):
        raise ValueError('置信度输入必须按正式相关性排序')
    return values


def snapshot(rows, *, retrieved, limit=5):
    if not retrieved:
        return None
    fields = (
        'id',
        'category',
        'questions',
        'answer',
        'section_path',
        'content_type',
        'score',
        'rerank_score',
    )
    return [deepcopy({k: row[k] for k in fields if k in row}) for row in rows[:limit]]


def assess(rows, profile):
    values = scores(rows)
    top1 = values[0] if values else 0.0
    gap = max(0.0, top1 - (values[1] if len(values) > 1 else 0.0))
    valid = sum(v >= profile['effective_cutoff'] for v in values)
    features = [
        min(1.0, max(0.0, top1)),
        min(1.0, valid / profile['count_cap']),
        min(1.0, gap),
    ]
    score = sum(w * x for w, x in zip(profile['weights'], features))
    return dict(
        evidence_confidence=score,
        allowed=bool(rows) and score >= profile['threshold'],
        top1=top1,
        top1_top2_gap=gap,
        effective_count=valid,
        profile_version=profile['version'],
    )


def confusion(rows, profile):
    counts = dict(tp=0, fp=0, tn=0, fn=0)
    for row in rows:
        found = {r['id'] for r in row['retrieval']['ranked']}
        relevant = set(row['relevant_ids'])
        label = bool(relevant) and relevant <= found
        predicted = assess(row['retrieval']['ranked'], profile)['allowed']
        counts[
            ('tp' if label else 'fp') if predicted else ('fn' if label else 'tn')
        ] += 1
    tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    counts.update(
        precision=precision,
        recall=recall,
        f_beta_05=(
            1.25 * precision * recall / (0.25 * precision + recall)
            if precision + recall
            else 0.0
        ),
    )
    return counts


def calibrate(cases):
    if len(cases) < 10 or any('error' in r for r in cases):
        raise ValueError('校准需至少10条完整标注与真实排名')
    # 按bucket固定哈希分层抽验证，不读取验证分数选择参数。
    train, validation = [], []
    for bucket in sorted({r['bucket'] for r in cases}):
        group = sorted(
            (r for r in cases if r['bucket'] == bucket),
            key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest(),
        )
        size = max(1, len(group) // 5)
        validation.extend(group[:size])
        train.extend(group[size:])
    if not train:
        raise ValueError('没有可用训练分区')
    observed = sorted({v for r in train for v in scores(r['retrieval']['ranked'])})
    if not observed:
        raise ValueError('训练分区没有重排结果')
    # 候选有效门槛取训练观测分位，所有特征权重非零；优化偏重精确率的F0.5。
    cutoffs = sorted(
        {
            observed[min(len(observed) - 1, int((len(observed) - 1) * q / 10))]
            for q in range(1, 10)
        }
    )
    best = None
    for cutoff in cutoffs:
        for cap in (1, 2, 3, 5):
            for weights in (
                [0.6, 0.3, 0.1],
                [0.7, 0.2, 0.1],
                [0.5, 0.3, 0.2],
                [0.8, 0.1, 0.1],
            ):
                profile = dict(
                    version='ch09-grid-v1',
                    effective_cutoff=cutoff,
                    count_cap=cap,
                    weights=weights,
                    threshold=0.0,
                )
                values = sorted(
                    {
                        assess(r['retrieval']['ranked'], profile)['evidence_confidence']
                        for r in train
                    }
                )
                for threshold in values + [max(values) + 1e-9]:
                    candidate = {**profile, 'threshold': threshold}
                    metrics = confusion(train, candidate)
                    key = (
                        metrics['f_beta_05'],
                        metrics['precision'],
                        metrics['recall'],
                        threshold,
                    )
                    if best is None or key > best[0]:
                        best = key, candidate
    profile = best[1]
    report = dict(
        train_ids=sorted(r['id'] for r in train),
        validation_ids=sorted(r['id'] for r in validation),
        train=confusion(train, profile),
        validation=confusion(validation, profile),
        limitation='开发集内固定分层验证，不是独立留出集；经验分不是概率，不能证明事实覆盖',
    )
    return profile, report
