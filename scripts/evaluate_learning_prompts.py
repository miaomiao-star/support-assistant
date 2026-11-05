"""Ch09标准化、语义去重、充分性自评的真实标注验证；不访问业务库。"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.llm import get_chat_model
from app.knowledge.mining import structured
from app.learning.prompts import (
    FaqDraft,
    SemanticMatch,
    Sufficiency,
    NORMALIZE_PROMPT,
    DEDUPE_PROMPT,
    SUFFICIENCY_PROMPT,
)
from app.workflow.persistence import run_async


async def main():
    cases = json.loads(
        (ROOT / 'tests/fixtures/ch09_prompt_cases.json').read_text(encoding='utf-8')
    )
    path = ROOT / 'dev-notes' / f'ch09-prompts-{time.time_ns()}.json'
    rows = []
    model = get_chat_model()
    for case in cases:
        row = dict(id=case['id'])
        try:
            if case['kind'] == 'normalize':
                result = await structured(
                    model,
                    FaqDraft,
                    NORMALIZE_PROMPT,
                    dict(question=case['question'], retrieved_chunks=case['chunks']),
                )
                passed = all(
                    token in result.normalized_question for token in case['must_keep']
                ) and (not case['unverified'] or '待人工' in result.ai_suggested_answer)
            elif case['kind'] == 'dedupe':
                result = await structured(
                    model,
                    SemanticMatch,
                    DEDUPE_PROMPT,
                    dict(question=case['question'], candidates=case['candidates']),
                )
                passed = result.matched_review_id == case['expected']
            else:
                result = await structured(
                    model,
                    Sufficiency,
                    SUFFICIENCY_PROMPT,
                    {k: case[k] for k in ('question', 'draft', 'evidence')},
                )
                passed = result.useful == case['expected']
            row.update(result=result.model_dump(), passed=passed)
        except Exception as exc:
            row.update(passed=False, error=type(exc).__name__)
        rows.append(row)
        path.write_text(
            json.dumps(
                dict(
                    results=rows,
                    passed=sum(r['passed'] for r in rows),
                    total=len(cases),
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        print(('PASS' if row['passed'] else 'FAIL') + ' ' + case['id'], flush=True)
    print(f"{sum(r['passed'] for r in rows)}/{len(cases)} {path}")
    return all(r['passed'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(0 if run_async(main()) else 1)
