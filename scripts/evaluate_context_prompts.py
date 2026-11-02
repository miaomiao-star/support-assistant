"""标注样例检验事实摘要，纯Prompt任务用真实输出替代TDD。"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.core.llm import get_chat_model
from app.context.summary import summarize_batch
from app.workflow.persistence import run_async

# 这些是开发标注样例，不是独立留出集；电话为固定演示值，没有导入用户通讯录。
CASES = [
    dict(
        name='订单诉求未完成',
        messages=[
            HumanMessage(content='订单1001的MH-W40饮水机漏水，我想退货。'),
            AIMessage(content='需提供照片核实，目前没有批准或执行退款。'),
        ],
        background=[],
        must=['1001', 'MH-W40', '漏水', '退货'],
        forbid=['已退款', '已批准'],
    ),
    dict(
        name='明确联系方式',
        messages=[
            HumanMessage(content='订单1002想维修，演示联系电话18800001111。'),
            AIMessage(content='已记录诉求，维修尚未处理。'),
        ],
        background=[],
        must=['1002', '18800001111', '维修'],
        forbid=['维修完成', '已经修好'],
    ),
    dict(
        name='否定保留',
        messages=[
            HumanMessage(content='不是退货，订单1003只是查物流，还没有收到。'),
            AIMessage(content='模拟物流显示运输中，不能确认实际送达。'),
        ],
        background=[],
        must=['1003', '物流'],
        forbid=['已收到', '申请退货', '已签收'],
    ),
    dict(
        name='闲聊为空',
        messages=[
            HumanMessage(content='你好呀'),
            AIMessage(content='你好，有什么可以帮你？'),
            HumanMessage(content='谢谢，再见'),
        ],
        background=[],
        must=[],
        forbid=[],
        empty=True,
    ),
    dict(
        name='旧段不重复压缩',
        messages=[
            HumanMessage(content='现在只查订单1002的物流'),
            AIMessage(content='模拟查询显示订单1002已签收。'),
        ],
        background=['订单1001的MH-W40饮水机漏水，用户申请退货，未处理。'],
        must=['1002', '签收'],
        forbid=['1001', 'MH-W40', '漏水', '工具结果', '工具确认'],
    ),
    dict(
        name='工具事实和不确定性',
        messages=[
            HumanMessage(content='订单1004有没有发出？'),
            ToolMessage(
                content='{"ok":true,"order_id":"1004","status":"待发货","mock":true}',
                tool_call_id='demo',
            ),
            AIMessage(content='模拟结果待发货，实际需以平台为准。'),
        ],
        background=[],
        must=['1004', '待发货', '模拟'],
        forbid=['已经发货', '已送达'],
    ),
]


async def main():
    """调用真实摘要Prompt并存完整输出；词项断言不替代事实来源/否定的人工检查。"""
    model = get_chat_model()
    results = []
    for case in CASES:
        start = time.monotonic()
        content = await summarize_batch(model, case['messages'], case['background'])
        passed = (
            # 同时检查应保留词、禁止事实和无事实空串；保留content便于发现词项未覆盖的幻觉。
            all(term in content for term in case['must'])
            and not any(term in content for term in case['forbid'])
            and (not content if case.get('empty') else bool(content))
        )
        results.append(
            dict(
                name=case['name'],
                content=content,
                passed=passed,
                elapsed_ms=round((time.monotonic() - start) * 1000),
            )
        )
        print(case['name'], 'PASS' if passed else 'FAIL', flush=True)
    path = ROOT / 'dev-notes' / f'ch07-summary-eval-{time.time_ns()}.json'
    path.write_text(
        json.dumps(
            dict(
                passed=sum(row['passed'] for row in results),
                total=len(results),
                results=results,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    print(path, flush=True)
    if not all(row['passed'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    run_async(main())
