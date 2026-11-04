"""区分业务查询和政策判断的真实标注样例，不执行业务工具。"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.workflow.tooling import OperationResult, OPERATION_PROMPT
from app.workflow.understanding import json_call
from app.core.llm import get_chat_model
from app.workflow.persistence import run_async


async def main():
    # 这是真实上游标注验证，不能当隔离单测；只分类，不执行MCP或创建业务数据。
    cases = json.loads(
        (ROOT / 'tests/fixtures/ch08_operation_cases.json').read_text(encoding='utf-8')
    )
    report = {'results': []}
    path = ROOT / 'dev-notes' / f'ch08-operations-{time.time_ns()}.json'
    # 每次独立报告、每例即时落盘，保留此前失败；样例通过不代表独立留出集准确率。
    for case in cases:
        row = dict(case=case)
        try:
            result = await json_call(
                get_chat_model(),
                OperationResult,
                OPERATION_PROMPT,
                {'question': case['question'], 'history': []},
                256,
            )
            row.update(
                # intent与operation都对才通过，防止“售后”对了却误走政策资格闸。
                result=result.model_dump(),
                passed=result.intent == case['intent']
                and result.operation == case['operation'],
            )
        except Exception as exc:
            row.update(passed=False, error=type(exc).__name__)
        report['results'].append(row)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        print(
            ('PASS' if row['passed'] else 'FAIL') + ' ' + case['question'], flush=True
        )
    report.update(passed=sum(r['passed'] for r in report['results']), total=len(cases))
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} {path}")
    return report['passed'] == report['total']


if __name__ == '__main__':
    raise SystemExit(0 if run_async(main()) else 1)
