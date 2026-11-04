"""本章建单Prompt的真实开发样例验证，不执行业务工具。"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from langchain_core.messages import SystemMessage, HumanMessage
from app.workflow.tooling import SYSTEM
from app.tools.runtime import ToolRuntime
from app.core.llm import get_chat_model
from app.workflow.persistence import run_async


async def main():
    # 只绑定Schema检查真实模型如何追问/申请，不执行工具；本命令会产生模型调用费用。
    tools = [s.model_schema() for s in ToolRuntime().catalog.snapshot().values()]
    model = (
        get_chat_model()
        .bind_tools(tools, parallel_tool_calls=False)
        .bind(max_tokens=600)
    )
    cases = json.loads(
        (ROOT / 'tests/fixtures/ch08_prompt_cases.json').read_text(encoding='utf-8')
    )
    report = {'results': []}
    path = ROOT / 'dev-notes' / f'ch08-prompts-{time.time_ns()}.json'
    # 每例按标注检查是否应申请及描述来源；这不能代替后端执行权限或浏览器验收。
    for case in cases:
        row = dict(case=case['name'])
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content=SYSTEM),
                    HumanMessage(content=case['question'], name='current_question'),
                    HumanMessage(
                        content=json.dumps(
                            dict(
                                mode=(
                                    'ticket_intake'
                                    if case['authorized']
                                    else 'business'
                                ),
                                ticket_sources=case['sources'],
                                evidence=[],
                                confirmed_order=None,
                            ),
                            ensure_ascii=False,
                        ),
                        name='context_bundle',
                    ),
                ]
            )
            calls = [
                dict(name=c['name'], args=c['args'], id=c['id'])
                for c in response.tool_calls
            ]
            tickets = [c for c in calls if c['name'] == 'create_ticket']
            passed = bool(tickets) == case['expect_tool']
            if case['expect_tool']:
                # 描述必须与标注客户原话一致，不能只因“调用了建单工具”就算通过。
                passed = (
                    passed
                    and len(tickets) == 1
                    and tickets[0]['args'].get('description') == case['description']
                )
            if case.get('must_ask'):
                passed = passed and any(
                    w in response.content for w in ('问题', '情况', '描述', '说明')
                )
            row.update(passed=passed, answer=response.content, calls=calls)
        except Exception as exc:
            row.update(passed=False, error=type(exc).__name__)
        report['results'].append(row)
        # 增量保存当前结果，重新评估也不覆盖先前失败报告。
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(('PASS' if row['passed'] else 'FAIL') + ' ' + case['name'], flush=True)
    report.update(passed=sum(r['passed'] for r in report['results']), total=len(cases))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} {path}")
    return report['passed'] == report['total']


if __name__ == '__main__':
    raise SystemExit(0 if run_async(main()) else 1)
