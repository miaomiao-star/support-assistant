"""现有ChatOpenAI的窄协议适配：保留上游工具循环所需字段，不改变模型模式。"""

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class CompatibleChatOpenAI(ChatOpenAI):
    """对应langchain-openai1.7.1的转换扩展点；升级时须重跑真实SDK载荷测试。"""

    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ):
        """保留流式协议片段，仍只让content承载用户可见正文。"""
        generation = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if generation is not None and choices:
            value = (choices[0].get("delta") or {}).get("reasoning_content")
            if isinstance(value, str):
                generation.message.additional_kwargs["reasoning_content"] = value
        return generation

    def _create_chat_result(self, response, generation_info=None):
        """非流式JSON调用也保留供应商字段，不把它混进结构化结果正文。"""
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, data.get("choices") or []):
            value = choice["message"].get("reasoning_content")
            if isinstance(value, str):
                generation.message.additional_kwargs["reasoning_content"] = value
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        """在最终发送边界回填字段；不修改原始消息、数据库审计或checkpoint。"""
        messages = self._convert_input(input_).to_messages()
        last_user = max(
            (
                index
                for index, message in enumerate(messages)
                if message.type == "human"
            ),
            default=-1,
        )
        skipped_calls = set()
        retained = []
        for index, message in enumerate(messages):
            # 调用ID可能跨轮重用，旧轮审计的省略范围不能越过新的用户消息。
            if message.type == "human":
                skipped_calls.clear()
            # 旧MySQL/后端确认审计没有模型协议字段，不能伪造。仅去掉已完成的旧工具链。
            # 当前轮模型工具申请必须保留，若缺字段仍应暴露错误而不是暗中丢弃。
            if (
                index < last_user
                and isinstance(message, AIMessage)
                and message.tool_calls
                and not isinstance(
                    message.additional_kwargs.get("reasoning_content"), str
                )
            ):
                skipped_calls.update(call["id"] for call in message.tool_calls)
                continue
            if message.type == "tool" and message.tool_call_id in skipped_calls:
                continue
            retained.append(message)
        payload = super()._get_request_payload(retained, stop=stop, **kwargs)
        for message, wire in zip(retained, payload.get("messages", [])):
            if isinstance(message, AIMessage):
                value = message.additional_kwargs.get("reasoning_content")
                if isinstance(value, str):
                    wire["reasoning_content"] = value
        return payload
