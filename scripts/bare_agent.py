"""先看最裸循环：默认离线协议演示，不调用模型、不创建业务数据。"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.workflow.bare import run_loop


async def demo():
    """用固定模型响应和模拟观察展示循环；真实HTTP入口见同文件live函数。"""
    # 前两次申请工具、第三次直接作答，让第一次阅读能看清闭环而不受模型波动干扰。
    replies = iter(
        [
            dict(
                tool_calls=[
                    dict(
                        id="order",
                        type="function",
                        function=dict(
                            name="query_order", arguments='{"order_id":"1001"}'
                        ),
                    )
                ]
            ),
            dict(
                tool_calls=[
                    dict(
                        id="logistics",
                        type="function",
                        function=dict(
                            name="query_logistics", arguments='{"order_id":"1001"}'
                        ),
                    )
                ]
            ),
            dict(content="已先读取订单观察，再读取物流观察；此处是离线协议演示。"),
        ]
    )

    async def ask(messages):
        """离线替身展示上一次观察，再返回预先准备的一次模型响应。"""
        if messages[-1]["role"] == "tool":
            print("喂回 LLM 的观察：", messages[-1]["content"])
        response = next(replies)
        print("LLM 返回：", json.dumps(response, ensure_ascii=False))
        return response

    async def execute(call):
        """离线观察替身只打印工具名和演示数据，不访问任何业务表。"""
        print("执行工具：", call["name"])
        return dict(ok=True, data=dict(mock=True, order_id=call["args"]["order_id"]))

    result = await run_loop(
        [dict(role="user", content="先查订单1001，再查物流并汇总")],
        ask,
        execute,
        {"query_order", "query_logistics"},
    )
    print("收敛答案：", result["answer"])
    print("模型决策次数：", result["steps"])


async def live(question):
    """HTTP 原始 JSON 调用现有兼容上游；循环仍不使用模型/编排框架。"""
    import httpx
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from app.config import get_settings
    from app.tools.registry import ToolRegistry

    settings = get_settings()
    # 这三个工具完全是随机演示数据，不需要repo；FAQ/工单不会被allowed集合放行。
    registry = ToolRegistry(None, 0, question, timeout=settings.tool_timeout_seconds)
    allowed = {"query_order", "query_logistics", "query_product"}
    schemas = [
        convert_to_openai_tool(tool)
        for name, tool in registry.tools.items()
        if name in allowed
    ]
    # 适配层只复用 Ch02 的三个模拟只读工具，不创建会话或工单。
    async with httpx.AsyncClient(timeout=60) as client:

        async def ask(messages):
            """按Chat Completions原始协议发HTTP，不调用LangChain模型封装。"""
            response = await client.post(
                settings.chat_base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": "Bearer "
                    + settings.chat_api_key.get_secret_value()
                },
                json=dict(
                    model=settings.chat_model,
                    temperature=0,
                    max_tokens=600,
                    messages=messages,
                    tools=schemas,
                    parallel_tool_calls=False,
                ),
            )
            response.raise_for_status()
            # 只抽取assistant协议消息交给裸循环；API响应头、密钥和地址不打印。
            message = response.json()["choices"][0]["message"]
            names = [call["function"]["name"] for call in message.get("tool_calls", [])]
            print("LLM 决策：", names or "输出最终答案", flush=True)
            return message

        result = await run_loop(
            [
                dict(
                    role="system",
                    content="你是客服。工具均是随机模拟数据，回答必须明确说明。按用户要求先查订单再查物流，不猜订单号；缺号就追问。",
                ),
                dict(role="user", content=question),
            ],
            ask,
            registry.execute,
            allowed,
        )
    print("工具顺序：", " → ".join(result["trace"]))
    print("决策次数：", result["steps"])
    print("回答：", result["answer"])


if __name__ == "__main__":
    # 默认离线；只有显式--live才会使用真实上游，方便安全地先理解循环本身。
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="真实调用现有上游，可能产生费用"
    )
    parser.add_argument(
        "--question", default="先查订单1001的状态，再查这个订单的物流，最后汇总两项结果"
    )
    args = parser.parse_args()
    try:
        asyncio.run(live(args.question) if args.live else demo())
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        print("演示未完成：", type(exc).__name__, "HTTP", status, file=sys.stderr)
        raise SystemExit(1)
