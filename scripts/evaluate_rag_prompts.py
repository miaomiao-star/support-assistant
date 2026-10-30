"""纯Prompt任务使用标注样例核验，自评、型号、冲突、引用和负面承诺。"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.llm import get_chat_model
from app.rag.quality import GroundedGenerator


def matches_case(case, outcome):
    """标注断言允许等价否定表达，不能因“无法/不能”措辞差异误判质量。"""
    answer = outcome['answer']
    return (
        outcome['useful'] == case['useful']
        and all(w in answer for w in case.get('contains', []))
        and (
            not case.get('any_contains')
            or any(w in answer for w in case['any_contains'])
        )
    )


async def main():
    # 这是固定证据的Prompt测试，不检索向量库，专门隔离充分性/引用/负面承诺问题。
    cases = json.loads(
        (ROOT / 'tests/fixtures/ch04_prompt_cases.json').read_text(encoding='utf-8')
    )
    generator = GroundedGenerator(get_chat_model())
    results = []
    for c in cases:
        rows = (
            [
                {
                    'id': 1,
                    'questions': '标注样例',
                    'answer': c['answer'],
                    'section_path': '标注样例 / 证据1',
                }
            ]
            if c['answer']
            else []
        )
        if c.get('extra'):
            rows.append(
                {
                    'id': 2,
                    'questions': '标注样例',
                    'answer': c['extra'],
                    'section_path': '标注样例 / 证据2',
                }
            )
        # 例如extra可提供相互冲突的第二条证据，测试模型是否会拒绝强行作答。
        outcome = await generator.generate(
            c['query'], {'ranked': rows, 'low_confidence': not rows}
        )
        passed = matches_case(c, outcome)
        results.append(
            {
                'id': c['id'],
                'expected_useful': c['useful'],
                'passed': passed,
                'outcome': outcome,
            }
        )
        print(c['id'], passed, flush=True)
    report = {'passed': all(r['passed'] for r in results), 'results': results}
    path = ROOT / 'dev-notes' / f'ch04-prompts-{time.time_ns()}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(path, flush=True)


if __name__ == '__main__':
    asyncio.run(main())
