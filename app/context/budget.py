"""把可发送历史从模型窗口倒推；字符口径与预算共享同一配置。"""

import json
import math
from dataclasses import dataclass


def text_tokens(text, settings):
    """按当前项目的字符口径估算文本，结果用于预算而不是供应商计费。"""
    text = str(text)
    # CJK字符用中文系数，其余字符（含标点/数字）统一走另一系数；不是精确分词器。
    chinese = sum('\u3400' <= char <= '\u9fff' for char in text)
    return math.ceil(
        chinese / settings.zh_chars_pre_token
        + (len(text) - chinese) / settings.context_ascii_chars_per_token
    )


def count_tokens(messages, settings):
    """同时数协议和隐藏推理长度，但日志不输出隐藏推理正文。"""
    total = 0
    for message in messages:
        # 每条消息另计角色/协议包装的经验开销，不能只数用户和客服的可见正文。
        total += 6 + text_tokens(message.content, settings)
        if getattr(message, 'tool_calls', None):
            total += text_tokens(
                json.dumps(message.tool_calls, ensure_ascii=False), settings
            )
        total += text_tokens(
            message.additional_kwargs.get('reasoning_content', '') or '', settings
        )
    return total


@dataclass(frozen=True)
class ContextBudget:
    # capacity是窗口扣除本轮预留后的容量；history还会受“想保留多少轮”上限限制。
    # frozen防止某个节点临时放大共享预算；所有调用都从同一Settings重新推导。
    history: int
    layer1: int
    layer2: int
    capacity: int
    react_peak: int

    @classmethod
    def from_settings(cls, settings):
        """先扣固定开销和本轮工具链峰值，再给历史分配七成/三成。"""
        # 最后一跳通常是最终答复，已由output预留；此前至多N-1组申请+观察需额外扣除。
        peak = max(settings.agent_max_steps - 1, 0) * (
            settings.context_intermediate_tokens + settings.tool_result_max_tokens
        )
        capacity = (
            settings.model_context_window
            - settings.context_system_reserve
            - settings.rerank_top_k * settings.context_evidence_per_hit
            - settings.context_summary_budget
            - settings.agent_output_token_limit
            - settings.context_safety_margin
            - settings.max_user_input_tokens
            - peak
        )
        # 大窗口也不无限注入旧历史；R×S表达产品想保留的量，capacity表达模型装得下的量。
        history = min(
            settings.context_target_rounds * settings.context_steady_round_tokens,
            capacity,
        )
        if history < settings.context_steady_round_tokens:
            # 这是启动配置错误；不能悄悄把滑窗改成0并宣称上下文管理正常。
            raise ValueError(
                '上下文预算不足：固定开销和单轮峰值后连一轮稳态历史都装不下'
            )
        # 保留本课Python浮点int口径：5650对应3954/1695，剩余1 token不分配。
        return cls(history, int(history * 0.7), int(history * 0.3), capacity, peak)
