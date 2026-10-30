"""可单测检索指标；无答案样例单列拒答统计，不参与正例Recall平均。"""


def mean_or_none(values):
    """空样本保留N/A，不能当作零分或满分混入评估平均。"""
    if not values:
        return None
    return sum(values) / len(values)


def retrieval_metrics(rows, relevant, ks=(1, 5, 10)):
    if not relevant:
        return None
    ids = [r['id'] for r in rows]
    # Recall分母是全部相关块；多证据问题只找回其中一块不能算全部召回。
    result = {
        f'recall@{k}': len(set(ids[:k]) & set(relevant)) / len(relevant) for k in ks
    }
    # MRR只关心首个相关结果的位置：第1条为1、第2条为1/2，前10无命中为0。
    result['mrr@10'] = next(
        (1 / (i + 1) for i, key in enumerate(ids[:10]) if key in relevant), 0.0
    )
    return result
