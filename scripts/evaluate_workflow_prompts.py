"""真实意图 Prompt 验证；不使用 mock，不输出模型地址或密钥。"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.llm import get_chat_model
from app.workflow.graph import ROUTES
from app.workflow.intent import Intent, classify
from app.workflow.persistence import run_async


async def main():
    """逐条运行人工标注的意图样例，保存预测、路由、耗时与失败，不执行Agent。"""
    cases = json.loads(
        (ROOT / "tests/fixtures/ch05_intents.json").read_text(encoding="utf-8")
    )
    # 先校验样例ID和标签，避免把坏标注当成模型分类失败。
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        Intent(intent=case["intent"])
        assert case["question"].strip()
    report = dict(kind="live_intent_prompt", total=len(cases), rows=[], completed=False)
    path = ROOT / "dev-notes" / f"ch05-prompts-{time.time_ns()}.json"
    model = get_chat_model()
    # 与线上共用classify函数，比较“期望意图”与真实输出，不用固定替身刷通过率。
    for case in cases:
        start = time.monotonic()
        try:
            predicted = await classify(model, case["question"])
            row = dict(
                **case,
                predicted=predicted,
                route=ROUTES[predicted],
                passed=predicted == case["intent"],
                elapsed=round(time.monotonic() - start, 3),
            )
            report["rows"].append(row)
            print(
                case["id"], predicted, "PASS" if row["passed"] else "FAIL", flush=True
            )
        except Exception as exc:
            # 额度/认证失败不会靠换端点解决，也不继续对其余样例重复请求。
            report["rows"].append(
                dict(
                    **case,
                    passed=False,
                    error=type(exc).__name__,
                    http_status=getattr(exc, "status_code", None),
                )
            )
            report["blocked"] = True
            print(
                case["id"],
                type(exc).__name__,
                "HTTP",
                getattr(exc, "status_code", None),
                flush=True,
            )
            break
        finally:
            # 每题都写报告；中途失败仍有已完成样例，时间戳文件不会覆盖历史报告。
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    # 完成遍历和全部正确是不同指标；退出码只有两者同时满足才为0。
    report["completed"] = len(report["rows"]) == len(cases) and not report.get(
        "blocked"
    )
    report["passed"] = sum(row["passed"] for row in report["rows"])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f'{report["passed"]}/{len(cases)}；报告 {path.name}')
    return 0 if report["completed"] and report["passed"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(run_async(main()))
