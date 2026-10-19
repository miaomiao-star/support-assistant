"""单轮工具编排：选择一次 → 执行最多一个 → 无工具模型流式收敛。"""
import asyncio
import json
from contextlib import aclosing

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.core.memory import BudgetExceeded, estimate_tokens, prepare_messages
from app.core.prompts import CUSTOMER_SERVICE_PROMPT, CUSTOMER_SERVICE_SYSTEM
from app.tools.registry import ToolRegistry


def text_delta(chunk):
    """
    不同上游可能返回字符串或内容块列表；这里只提取可显示的 text。
    一个 chunk 可能含多个字，也可能只有工具参数，不等于精确的一个 token。
    """
    if isinstance(chunk.content, str):
        return chunk.content
    return ''.join(b.get('text', '') for b in chunk.content
                   if isinstance(b, dict) and b.get('type') == 'text')


class ChatService:
    # service 决定“先做什么、后做什么”，不拼 SQL，也不直接生成 HTML/SSE 字符串。
    # repo 提供数据访问，model 提供模型调用；测试可以替换它们而不请求真实上游。
    def __init__(self, repo, model, text_budget, tool_budget, chars_per_token, tool_timeout):
        self.repo, self.model = repo, model
        self.text_budget, self.tool_budget = text_budget, tool_budget
        self.chars_per_token, self.tool_timeout = chars_per_token, tool_timeout

    async def stream(self, session_id, user_text):
        """
        这是异步生成器：每次 yield 交出一个内部事件，API 层负责包装 SSE。
        第 1 步：找到持久化会话，准备本轮专用的工具注册表。
        session_id 是浏览器会话标识；cid 是 conversations 表主键。
        """
        cid = await self.repo.open_conversation(session_id)
        registry = ToolRegistry(self.repo, cid, user_text, timeout=self.tool_timeout)
        tools = list(registry.tools.values())
        # 第 2 步：工具定义本身也会进入模型上下文，必须为它预留预算。
        # convert_to_openai_tool 生成工具名称、描述、参数 JSON Schema；
        # 此时只是在向模型介绍工具，没有执行任何业务函数。
        schemas = json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False)
        schema_cost = len(schemas) / self.chars_per_token
        budget = min(self.text_budget, int(self.tool_budget - schema_cost))
        # 历史以数据库为准；prepare_messages 保留 system、最近完整回合、当前输入。
        # min 同时满足原文本预算和扣除工具定义后的总预算，超额会抛异常。
        messages = prepare_messages(await self.repo.history(cid), user_text, CUSTOMER_SERVICE_SYSTEM,
                                    budget, self.chars_per_token)
        await self.repo.add_messages(cid, [HumanMessage(content=user_text)])
        # 用户输入先保存，后续断线也能审计；未闭合的回合不会用于下次模型输入。
        # prepare_messages 已包含 system；模板自身也有 system，所以从 [1:] 开始。
        prompt = await CUSTOMER_SERVICE_PROMPT.ainvoke({'history': messages[1:]})
        # 第 3 步：模型决定是否调用工具。bind_tools 返回绑定后的调用对象，
        # 不会执行工具，也不把“原模型”永久变成自动执行 Agent。
        # parallel_tool_calls=False 是请求给上游的约束；服务端稍后仍会检查。
        bound = self.model.bind_tools(tools, parallel_tool_calls=False)
        aggregate, parts = None, []
        # aggregate 合并完整协议消息（包括分段工具参数）；parts 仅积累文本。
        # 工具参数 JSON 可能跨多个 chunk，所以不能在首个参数片段到达时就执行。
        # aclosing 确保本生成器提前关闭时，也关闭内层上游异步流。
        async with aclosing(bound.astream(prompt)) as chunks:
            async for chunk in chunks:
                aggregate = chunk if aggregate is None else aggregate + chunk
                delta = text_delta(chunk)
                if delta:
                    # 普通聊天直接在这次模型流中回复；工具选择时若带解释文本，
                    # 也会显示出来，之后保存为带工具申请的 assistant 消息正文。
                    parts.append(delta)
                    yield dict(type='delta', delta=delta)
                await asyncio.sleep(0)
        if aggregate is None or aggregate.invalid_tool_calls:
            # 流结束后再检查工具协议是否可解析；错误申请不能进入业务执行。
            raise ValueError('Invalid model response')
        calls = aggregate.tool_calls
        if not calls:
            # 分支 A：无工具。第一条模型流就是本轮回答，不额外请求第二次模型。
            # 只有完整且非空的回答才落库；中途失败时不会保存半截助手回复。
            if not parts:
                raise ValueError('Empty model reply')
            await self.repo.add_messages(cid, [AIMessage(content=''.join(parts))])
            return
        # 分支 B：有工具。先校验 ID 的非空、长度和本轮唯一性。
        # 数据库 tool_call_id 长度为 64；重复 ID 无法可靠对应结果。
        if (any(not c.get('id') or len(c['id']) > 64 for c in calls)
                or len({c['id'] for c in calls}) != len(calls)):
            raise ValueError('Invalid tool call id')
        request_message = AIMessage(content=''.join(parts), tool_calls=calls)
        # 第 4 步：保存模型的“工具申请单”。anchor 是本条 messages 的自增主键，
        # 后续查询结果时以它为下界，防止上游跨回合重用 ID 时取到旧结果。
        anchor = await self.repo.add_messages(cid, [request_message])
        messages.append(request_message)
        for call in calls:
            # 第 5 步：先 yield running 再执行，浏览器能先显示工具状态徽章。
            # 这里的 for 只是逐一回应上游申请，不是 Agent Loop。
            # 若上游一次申请多个工具，全部返回 single_call_limit，均不执行。
            yield dict(type='tool', name=call['name'], call_id=call['id'], status='running')
            result = (dict(ok=False, error='single_call_limit') if len(calls) != 1
                      else await registry.execute(call))
            tool_message = await self.repo.record_tool_result(
                cid, call['id'], result, after_message_id=anchor)
            # create_ticket 已把工单和成功结果原子提交。即使执行器观察到超时，
            # 若仓库查到本次成功结果，就按已保存的事实回灌，避免误报“未创建”。
            effective = json.loads(tool_message.content)
            yield dict(type='tool', name=call['name'], call_id=call['id'],
                       status='success' if effective.get('ok') else 'error',
                       error=effective.get('error'))
            messages.append(tool_message)
        # 第 6 步：申请与结果作为一对加入模型输入，不能只传孤立的查询结果。
        # 第二次调用不绑定工具，不再携带工具定义；这里核对实际消息总预算。
        # 若结果过长，安全终止；不截断 JSON 或删除 tool 消息来勉强塞进去。
        if estimate_tokens(messages, self.chars_per_token) > self.tool_budget:
            raise BudgetExceeded('工具结果超过上下文预算')
        prompt = await CUSTOMER_SERVICE_PROMPT.ainvoke({'history': messages[1:]})
        final_parts = []
        # 原模型未绑定工具，服务端也不进入第二轮执行。
        # 第 7 步：让模型读取 ToolMessage 并组织最终回答，继续流式输出文本。
        # 因此“单轮工具调用”允许两次模型生成，并不等于只请求一次模型。
        async with aclosing(self.model.astream(prompt)) as chunks:
            async for chunk in chunks:
                if chunk.tool_calls or chunk.invalid_tool_calls:
                    # 即使上游异常地再次申请工具，也只报错，不执行第二轮业务。
                    raise ValueError('Unexpected second tool call')
                delta = text_delta(chunk)
                if delta:
                    final_parts.append(delta)
                    yield dict(type='delta', delta=delta)
                await asyncio.sleep(0)
        if not final_parts:
            raise ValueError('Empty final reply')
        # 第 8 步：完整回答落库。service 正常结束后，API 层才发送 [DONE]。
        # 如果在这里之前失败，已执行的工具流水/工单仍保留，半截最终回复不保存。
        await self.repo.add_messages(cid, [AIMessage(content=''.join(final_parts))])
