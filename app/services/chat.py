"""单轮工具编排：选择一次 → 执行最多一个 → 无工具模型流式收敛。"""

import asyncio
import json
import re
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
    return ''.join(
        b.get('text', '')
        for b in chunk.content
        if isinstance(b, dict) and b.get('type') == 'text'
    )


class ChatService:
    # service 决定“先做什么、后做什么”，不拼 SQL，也不直接生成 HTML/SSE 字符串。
    # repo 提供数据访问，model 提供模型调用；测试可以替换它们而不请求真实上游。
    def __init__(
        self,
        repo,
        model,
        text_budget,
        tool_budget,
        chars_per_token,
        tool_timeout,
        rag=None,
        filters=None,
    ):
        self.repo, self.model = repo, model
        self.text_budget, self.tool_budget = text_budget, tool_budget
        self.chars_per_token, self.tool_timeout = chars_per_token, tool_timeout
        # rag为应用共享的知识服务；filters与repo中的检索结果属于本轮请求。
        self.rag, self.filters = rag, filters or {}

    async def stream(self, session_id, user_text):
        """
        这是异步生成器：每次 yield 交出一个内部事件，API 层负责包装 SSE。
        第 1 步：找到持久化会话，准备本轮专用的工具注册表。
        session_id 是浏览器会话标识；conversation_id 是 conversations 表主键。
        """
        conversation_id = await self.repo.open_conversation(session_id)
        if self.rag:
            self.repo.raw_question = user_text
            self.repo.query_filters = self.filters
        # RAG工具还包含理解、嵌入和重排三个上游请求，沿用旧5秒预算会误报超时。
        registry = ToolRegistry(
            self.repo,
            conversation_id,
            user_text,
            timeout=max(60, self.tool_timeout) if self.rag else self.tool_timeout,
        )
        tools = list(registry.tools.values())
        # 第 2 步：工具定义本身也会进入模型上下文，必须为它预留预算。
        # convert_to_openai_tool 生成工具名称、描述、参数 JSON Schema；
        # 此时只是在向模型介绍工具，没有执行任何业务函数。
        schemas = json.dumps(
            [convert_to_openai_tool(t) for t in tools], ensure_ascii=False
        )
        schema_cost = len(schemas) / self.chars_per_token
        budget = min(self.text_budget, int(self.tool_budget - schema_cost))
        # 历史以数据库为准；prepare_messages 保留 system、最近完整回合、当前输入。
        # min 同时满足原文本预算和扣除工具定义后的总预算，超额会抛异常。
        messages = prepare_messages(
            await self.repo.history(conversation_id),
            user_text,
            CUSTOMER_SERVICE_SYSTEM,
            budget,
            self.chars_per_token,
        )
        await self.repo.add_messages(conversation_id, [HumanMessage(content=user_text)])
        # 用户输入先保存，后续断线也能审计；未闭合的回合不会用于下次模型输入。
        # prepare_messages 已包含 system；模板自身也有 system，所以从 [1:] 开始。
        prompt = await CUSTOMER_SERVICE_PROMPT.ainvoke({'history': messages[1:]})
        # 第 3 步：模型决定是否调用工具。bind_tools 返回绑定后的调用对象，
        # 不会执行工具，也不把“原模型”永久变成自动执行 Agent。
        # parallel_tool_calls=False 是请求给上游的约束；服务端稍后仍会检查。
        bound = self.model.bind_tools(tools, parallel_tool_calls=False)
        selection_message, selection_parts = None, []
        # selection_message 合并完整协议消息（包括分段工具参数）；selection_parts 仅积累文本。
        # 工具参数 JSON 可能跨多个 chunk，所以不能在首个参数片段到达时就执行。
        # aclosing 确保本生成器提前关闭时，也关闭内层上游异步流。
        async with aclosing(bound.astream(prompt)) as chunks:
            async for chunk in chunks:
                selection_message = (
                    chunk if selection_message is None else selection_message + chunk
                )
                delta = text_delta(chunk)
                if delta:
                    # 普通聊天直接在这次模型流中回复；工具选择时若带解释文本，
                    # 也会显示出来，之后保存为带工具申请的 assistant 消息正文。
                    selection_parts.append(delta)
                    # 知识模式暂存选工具时的解释文字，引用/充分性校验前不能提前显示。
                    if not self.rag:
                        yield dict(type='delta', delta=delta)
                await asyncio.sleep(0)
        if selection_message is None or selection_message.invalid_tool_calls:
            # 流结束后再检查工具协议是否可解析；错误申请不能进入业务执行。
            raise ValueError('Invalid model response')
        tool_calls = selection_message.tool_calls
        if (
            self.rag
            and len(tool_calls) == 1
            and tool_calls[0]['name'] == 'query_product'
        ):
            # 这是后端对工具能力的边界检查，不是对商品参数作推断。
            price_or_stock = any(
                w in user_text
                for w in ('价格', '多少钱', '售价', '库存', '有货', '余量')
            )
            specs = any(
                w in user_text
                for w in (
                    '容量',
                    '规格',
                    '功能',
                    '多大',
                    '尺寸',
                    '滤芯',
                    '支持',
                    '断电',
                    '清洗',
                    '联网',
                    '保修',
                )
            )
            if specs or not price_or_stock:
                # 模拟价格/库存工具没有规格证据，不能让错误选工具绕过拒答与入池。
                tool_calls = [
                    {
                        'name': 'query_faq',
                        'id': tool_calls[0]['id'],
                        'args': {'keyword': user_text[:64]},
                    }
                ]
                selection_parts = []
        # 启用知识护栏时，仅明确的寒暄允许无证据直答；其余漏选工具的输入仍须检索。
        # 合成一次原契约调用，不进入第二轮选工具，也不利用历史做指代消解。
        if (
            self.rag
            and not tool_calls
            and not re.fullmatch(
                r'\s*(你好|您好|嗨|谢谢|感谢|再见|拜拜|hello|hi)[!！。,.，\s]*',
                user_text,
                re.I,
            )
        ):
            tool_calls = [
                {
                    'name': 'query_faq',
                    'id': 'rag_fallback',
                    'args': {'keyword': user_text[:64]},
                }
            ]
            selection_parts = []
        if not tool_calls:
            # 分支 A：无工具。第一条模型流就是本轮回答，不额外请求第二次模型。
            # 只有完整且非空的回答才落库；中途失败时不会保存半截助手回复。
            if not selection_parts:
                raise ValueError('Empty model reply')
            if self.rag:
                for part in selection_parts:
                    yield dict(type='delta', delta=part)
            await self.repo.add_messages(
                conversation_id, [AIMessage(content=''.join(selection_parts))]
            )
            return
        # 分支 B：有工具。先校验 ID 的非空、长度和本轮唯一性。
        # 数据库 tool_call_id 长度为 64；重复 ID 无法可靠对应结果。
        if any(not c.get('id') or len(c['id']) > 64 for c in tool_calls) or len(
            {c['id'] for c in tool_calls}
        ) != len(tool_calls):
            raise ValueError('Invalid tool call id')
        request_message = AIMessage(
            content=''.join(selection_parts), tool_calls=tool_calls
        )
        # 第 4 步：保存模型的“工具申请单”。anchor 是本条 messages 的自增主键，
        # 后续查询结果时以它为下界，防止上游跨回合重用 ID 时取到旧结果。
        request_message_id = await self.repo.add_messages(
            conversation_id, [request_message]
        )
        messages.append(request_message)
        for call in tool_calls:
            # 第 5 步：先 yield running 再执行，浏览器能先显示工具状态徽章。
            # 这里的 for 只是逐一回应上游申请，不是 Agent Loop。
            # 若上游一次申请多个工具，全部返回 single_call_limit，均不执行。
            yield dict(
                type='tool', name=call['name'], call_id=call['id'], status='running'
            )
            result = (
                dict(ok=False, error='single_call_limit')
                if len(tool_calls) != 1
                else await registry.execute(call)
            )
            tool_message = await self.repo.record_tool_result(
                conversation_id, call['id'], result, after_message_id=request_message_id
            )
            # create_ticket 已把工单和成功结果原子提交。即使执行器观察到超时，
            # 若仓库查到本次成功结果，就按已保存的事实回灌，避免误报“未创建”。
            tool_result = json.loads(tool_message.content)
            yield dict(
                type='tool',
                name=call['name'],
                call_id=call['id'],
                status='success' if tool_result.get('ok') else 'error',
                error=tool_result.get('error'),
            )
            messages.append(tool_message)
        if (
            self.rag
            and len(tool_calls) != 1
            and any(c['name'] == 'query_faq' for c in tool_calls)
        ):
            # 协议拒绝意味着根本没有知识证据，不再放任第二次模型自由生成政策。
            raise RuntimeError('知识工具申请不满足单次调用约束')
        if self.rag and len(tool_calls) == 1 and tool_calls[0]['name'] == 'query_faq':
            if not tool_result.get('ok') or self.repo.latest_retrieval is None:
                # 连接或工具错误不能伪装成知识不足，更不能走无证据生成。
                raise RuntimeError('知识检索未完成')
            outcome = await self.rag.generator.generate(
                user_text, self.repo.latest_retrieval
            )
            # 业务拒答才入池；超时、连接错误在上面/生成内部抛出，走SSE错误而不入池。
            if not outcome['useful']:
                await self.rag.quality.low_confidence(
                    conversation_id, user_text, outcome['source'], outcome['reason']
                )
            # citations是实际Prompt证据全集，cited才是正文引用的编号子集。
            yield dict(
                type='citations',
                citations=outcome['citations'],
                cited=outcome['cited'],
                useful=outcome['useful'],
            )
            await self.repo.add_messages(
                conversation_id, [AIMessage(content=outcome['answer'])]
            )
            yield dict(type='delta', delta=outcome['answer'])
            return
        if self.rag:
            # 非FAQ仍沿用旧业务工具生成；先前缓冲的工具说明到此才允许展示。
            for part in selection_parts:
                yield dict(type='delta', delta=part)
        # 第 6 步：申请与结果作为一对加入模型输入，不能只传孤立的查询结果。
        # 第二次调用不绑定工具，不再携带工具定义；这里核对实际消息总预算。
        # 若结果过长，安全终止；不截断 JSON 或删除 tool 消息来勉强塞进去。
        if estimate_tokens(messages, self.chars_per_token) > self.tool_budget:
            raise BudgetExceeded('工具结果超过上下文预算')
        prompt = await CUSTOMER_SERVICE_PROMPT.ainvoke({'history': messages[1:]})
        answer_parts = []
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
                    answer_parts.append(delta)
                    yield dict(type='delta', delta=delta)
                await asyncio.sleep(0)
        if not answer_parts:
            raise ValueError('Empty final reply')
        # 第 8 步：完整回答落库。service 正常结束后，API 层才发送 [DONE]。
        # 如果在这里之前失败，已执行的工具流水/工单仍保留，半截最终回复不保存。
        await self.repo.add_messages(
            conversation_id, [AIMessage(content=''.join(answer_parts))]
        )
