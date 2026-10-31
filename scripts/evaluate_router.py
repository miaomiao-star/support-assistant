"""Ch06冻结标注评估；真实调用现有上游，失败报告逐次保留。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, AIMessage
from app.config import get_settings
from app.core.llm import get_chat_model
from app.tools.demo_orders import get_order
from app.workflow.persistence import run_async
from app.workflow.understanding import (
    understand,
    classify_intent,
    json_call,
    QueryExpansion,
    EXPAND_PROMPT,
)


async def evaluate(split):
    settings = get_settings()
    model = get_chat_model()
    data = json.loads(
        (ROOT / "tests/fixtures/ch06_eval.json").read_text(encoding="utf-8")
    )
    cases = [case for case in data["cases"] if split == "all" or case["split"] == split]
    path = ROOT / "dev-notes" / f"ch06-prompts-{time.time_ns()}.json"
    report = dict(
        model=settings.chat_model,
        dataset_version=data["version"],
        split=split,
        threshold=settings.intent_confidence_threshold,
        results=[],
    )
    semaphore = asyncio.Semaphore(3)

    async def one(case):
        async with semaphore:
            start = time.monotonic()
            output = dict(
                id=case["id"],
                split=case["split"],
                question=case["question"],
                expected_intent=case["intent"],
            )
            try:
                history = [
                    (HumanMessage if role == "human" else AIMessage)(content=text)
                    for role, text in case.get("history", [])
                ]
                result = await understand(model, case["question"], history, None)
                intent = await classify_intent(
                    model,
                    result.normalized_question,
                    settings.intent_confidence_threshold,
                )
                issues = []
                if intent.intent != case["intent"]:
                    issues.append("intent")
                if (
                    case.get("unchanged")
                    and result.resolved_question != case["question"]
                ):
                    issues.append("unchanged")
                if any(
                    token not in result.resolved_question
                    for token in case.get("contains", [])
                ):
                    issues.append("reference")
                if any(
                    token in result.resolved_question
                    for token in case.get("absent", [])
                ):
                    issues.append("ambiguous_reference")
                if "faq" in case and result.is_simple_faq != case["faq"]:
                    issues.append("faq")
                output.update(result.model_dump(), **intent.model_dump())
                if case.get("expand"):
                    expanded = await json_call(
                        model,
                        QueryExpansion,
                        EXPAND_PROMPT,
                        dict(
                            question=result.normalized_question,
                            order=get_order("1001"),
                            intent=intent.intent,
                        ),
                        450,
                    )
                    output["expansion"] = expanded.model_dump()
                output.update(passed=not issues, issues=issues)
            except Exception as exc:
                output.update(passed=False, error=type(exc).__name__)
            output["seconds"] = round(time.monotonic() - start, 3)
            report["results"].append(output)
            report["passed"] = sum(row["passed"] for row in report["results"])
            report["total"] = len(cases)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(case["id"], "PASS" if output["passed"] else "FAIL", flush=True)

    await asyncio.gather(*(one(case) for case in cases))
    print(f"{report['passed']}/{report['total']} {path}")
    return report["passed"] == report["total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    args = parser.parse_args()
    raise SystemExit(0 if run_async(evaluate(args.split)) else 1)
