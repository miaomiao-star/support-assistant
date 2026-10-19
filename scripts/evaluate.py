"""真实模型验收：先启动服务，再 uv run python scripts/evaluate.py。"""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import requests


ROOT = Path(__file__).resolve().parent.parent


# 与单测不同：这个函数真实请求本机服务，间接调用上游模型。
def chat(base_url: str, session_id: str, message: str) -> dict:
    chunks = []
    arrival_seconds = []
    # monotonic 测耗时，不受系统时钟调整影响。
    started = time.monotonic()
    done = False
    with requests.post(
        f"{base_url}/api/chat", json={"session_id": session_id, "message": message},
        stream=True, timeout=(10, 150),
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        # chunk_size=1 减少客户端读取缓冲；此处按行读取，不是每字节一个模型 token。
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                break
            data = json.loads(payload)
            if data.get("error") and "message" in data:
                raise RuntimeError(data["message"])
            # Ch02 新增工具状态事件；旧验收只统计最终文本增量。
            if "delta" not in data:
                continue
            chunks.append(data["delta"])
            arrival_seconds.append(round(time.monotonic() - started, 3))
    if not done or not chunks:
        raise RuntimeError("SSE 回复缺少内容或 DONE")
    return {"message": message, "reply": "".join(chunks), "chunk_count": len(chunks),
            "arrival_seconds": arrival_seconds, "done": done}


# 五条样例检查字段及方案关键词；关键词通过不等于完整语义评估。
# 二轮问题不重复商品名称，以检验服务端确实带上了历史。
def evaluate(base_url: str) -> dict:
    cases = json.loads((ROOT / "tests/fixtures/extract_cases.json").read_text(encoding="utf-8"))
    results = []
    for case in cases:
        response = requests.post(f"{base_url}/api/extract", json={"text": case["text"]}, timeout=(10, 150))
        response.raise_for_status()
        ticket = response.json()
        fields_ok = all(ticket.get(key) == value for key, value in case["expected"].items())
        solution = ticket.get("expected_solution", "")
        solution_ok = isinstance(solution, str) and bool(solution.strip()) and all(
            word in solution for word in case["solution_keywords"]
        )
        results.append({**case, "actual": ticket, "passed": fields_ok and solution_ok})
    # 每次评估使用独立会话，避免上次评估历史干扰。
    session = f"eval-{uuid4().hex}"
    first = chat(base_url, session, "我想咨询的商品名称是「云朵猫爬架」。请记住这个名字，不需要查询商品信息，简短确认即可。")
    second = chat(base_url, session, "我上一轮提到的商品是什么？只回答商品名称。")
    context_ok = "云朵猫爬架" in second["reply"]
    return {"extract": results, "conversation": [first, second], "context_passed": context_ok,
            "passed": all(item["passed"] for item in results) and context_ok and first["chunk_count"] > 1}


# 只有直接运行脚本才执行验收；被导入时不自动请求模型。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report = evaluate(args.base_url.rstrip("/"))
    output = ROOT / "dev-notes/ch01-evaluation.json"
    # 开发记录目录不入库，全新克隆运行评估时需要先创建目录。
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
