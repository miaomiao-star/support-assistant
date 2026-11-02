"""纯投影：只改变两个边界和发送副本，绝不删除checkpoint中的原文。"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, trim_messages

from app.context.budget import count_tokens


def anchor(message):
    """取SQL边界坐标：普通消息用自己的行ID，工具观察借用所属申请的行ID。"""
    return int(
        message.additional_kwargs.get(
            'sql_msg_id', message.additional_kwargs.get('anchor_id', 0)
        )
    )


def turns(messages):
    """按Human起点分组；一组可包含多次AI申请/Tool观察，不等于简单的一问一答。"""
    groups = []
    for message in messages:
        if isinstance(message, HumanMessage) or not groups:
            groups.append([])
        groups[-1].append(message)
    return groups


def shorten(messages, settings):
    """生成层2副本：保留用户原话，只缩客服正文及工具表示。"""
    result = []
    for message in messages:
        if message.type == 'tool':
            # 原始工具JSON还在State中；这里用普通AI历史标识避免发送残缺的原生工具协议。
            result.append(
                AIMessage(content=f'[工具结果 {message.name or message.tool_call_id}]')
            )
        elif message.type == 'ai':
            # 只保留开头，丢掉本次发送副本中的tool_calls/隐藏推理，不改原消息对象。
            content = str(message.content)[: settings.context_assistant_prefix_chars]
            if message.tool_calls:
                content += (
                    '[工具申请 '
                    + ','.join(call['name'] for call in message.tool_calls)
                    + ']'
                )
            if content:
                result.append(
                    AIMessage(
                        content=content,
                        id=message.id,
                        additional_kwargs={'sql_msg_id': anchor(message)},
                    )
                )
        else:
            result.append(message.model_copy(deep=True))
    return result


@dataclass
class HistoryView:
    # layer2_tokens是裁掉溢出内容之前的用量，后台触发必须看它，不能看clipped后的小数值。
    # omitted_pending只表示本次输入暂未装入全部层2，不代表这些原文已经被摘要覆盖。
    layer1: list
    layer2: list
    layer1_from: int
    layer2_tokens: int
    omitted_pending: bool = False

    @property
    def messages(self):
        # 拼装顺序从较远的层2到最近的层1；本轮用户消息由模型装配函数另加。
        return [*self.layer2, *self.layer1]


def project_history(messages, *, summary_upto, layer1_from, budget, settings):
    """根据两个排除端点计算发送视图，并返回本次建议推进到的层1边界。"""
    # 以完整用户轮作为最小降级单位，工具链不能被边界切成两半。
    groups = [
        group
        for group in turns(messages)
        if max(map(anchor, group), default=0) > summary_upto
    ]
    boundary = max(summary_upto, layer1_from)
    recent = [group for group in groups if max(map(anchor, group)) > boundary]
    while (
        recent
        and count_tokens([message for group in recent for message in group], settings)
        > budget.layer1
    ):
        # 至少一轮若能装入总历史预算，可借尚未使用的L2额度，不能压最新原文。
        if len(recent) == 1:
            if count_tokens(recent[0], settings) > budget.history:
                raise ValueError(
                    '上下文预算不足：最近一轮原文超过全部历史额度，请调整窗口与保留量'
                )
            break
        # 只移除局部recent列表中的最老一整轮；消息还在调用方的完整State列表里。
        boundary = max(map(anchor, recent.pop(0)))
    layer1 = [message.model_copy(deep=True) for group in recent for message in group]
    middle = [
        group for group in groups if summary_upto < max(map(anchor, group)) <= boundary
    ]
    # summary_upto < 回合末ID <= boundary为层2；boundary自身不属于层1。
    layer2 = shorten([message for group in middle for message in group], settings)
    usage = count_tokens(layer2, settings)
    quota = min(budget.layer2, max(0, budget.history - count_tokens(layer1, settings)))
    # 若最近单轮借用了L2额度，这里同步收紧L2，防止两层合计超过history预算。
    # LangChain仅裁输入副本；按Human起点保留尾部整轮。待摘要原文仍在State。
    clipped = trim_messages(
        layer2,
        max_tokens=quota,
        token_counter=lambda items: count_tokens(items, settings),
        strategy='last',
        start_on='human',
        allow_partial=False,
    )
    return HistoryView(layer1, clipped, boundary, usage, len(clipped) != len(layer2))
