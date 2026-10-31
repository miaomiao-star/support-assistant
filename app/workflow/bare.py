"""热身：只用 Python/JSON 表达 Agent 循环，不依赖编排或模型框架。"""

import json
from math import ceil


class LoopStopped(ValueError):
    """达到边界时停止；不能把被拦截的工具申请继续交给执行器。"""


async def run_loop(
    messages, ask, execute, allowed_tools, max_steps=6, max_tokens=24000
):
    """手写带工具的模型循环，返回答案、决策次数、完整消息与工具轨迹。

    messages 是原始协议字典列表；ask 负责调用模型，execute 负责适配既有工具。
    将两者作为参数传入，离线演示与真实HTTP可以共用同一段循环逻辑。
    """
    # 拷贝外层列表，避免追加协议消息时修改调用者传入的列表对象。
    messages = list(messages)
    used = 0
    seen = set()
    trace = []
    for step in range(1, max_steps + 1):
        # 每次重新发送的完整上下文都会消耗 token，不能只计新增文字。
        cost = ceil(len(json.dumps(messages, ensure_ascii=False)) / 1.2)
        if used + cost > max_tokens:
            raise LoopStopped("token_limit")
        used += cost
        # 第一步只请求模型；模型返回工具申请并不意味着工具已经执行。
        response = await ask(messages)
        used += ceil(len(json.dumps(response, ensure_ascii=False)) / 1.2)
        if used > max_tokens:
            raise LoopStopped("token_limit")
        calls = response.get("tool_calls") or []
        if not calls:
            # 没有工具申请就收敛；最终答案也可以是请求用户补充信息。
            answer = response.get("content") or ""
            if not answer.strip():
                raise LoopStopped("empty_answer")
            return dict(
                answer=answer,
                steps=step,
                messages=messages + [dict(role="assistant", content=answer)],
                trace=trace,
            )
        if len(calls) != 1:
            # 本章每次只执行一个工具；“多步”是观察回灌后的再次决策。
            raise LoopStopped("one_tool_per_step")
        call = calls[0]
        name = call["function"]["name"]
        if name not in allowed_tools or name == "create_ticket":
            raise LoopStopped("tool_not_allowed")
        args = json.loads(call["function"]["arguments"])
        # 原始API中的arguments是JSON字符串，转为字典后才交既有执行器做Schema校验。
        key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
        if not call.get("id") or key in seen or call["id"] in seen:
            raise LoopStopped("repeated_tool")
        seen.update((key, call["id"]))
        # 先保存 assistant 的申请，再追加匹配 ID 的工具观察，保持协议配对。
        messages.append(
            dict(role="assistant", content=response.get("content"), tool_calls=calls)
        )
        # 观察作为role=tool消息回灌，tool_call_id必须匹配刚才assistant的申请ID。
        result = await execute(dict(name=name, args=args, id=call["id"]))
        messages.append(
            dict(
                role="tool",
                tool_call_id=call["id"],
                content=json.dumps(result, ensure_ascii=False),
            )
        )
        trace.append(name)
    # 模型持续申请工具也不能无限跑；到此不再额外请求一次模型来解释停止原因。
    raise LoopStopped("step_limit")
