"""原样可见上下文写本地日志；密钥/请求头/上游隐藏推理不属于可见上下文。"""

import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from app.context.budget import text_tokens

logger = logging.getLogger('mewhelp.context')
scope = ContextVar('model_context_scope', default=None)
# ContextVar把“这次SDK调用属于哪个会话/步骤”隔离到当前协程，避免并发请求串日志。


def setup_logging(root):
    """为应用配置一个UTF-8文件输出；重复组装应用时不重复挂同一路径handler。"""
    path = Path(root) / 'log' / 'app.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == path.resolve()
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # 避免原文同时传播到通用控制台handler；此模块只输出调用方显式提供的字段。


def event(name, **values):
    """写一条可搜索的事件名及JSON；调用方负责不传凭据或完整配置对象。"""
    logger.info('%s %s', name, json.dumps(values, ensure_ascii=False, default=str))


@contextmanager
def capture_context(kind, settings, **metadata):
    """限定某次主力调用的打点范围，退出/异常时恢复上层作用域。"""
    token = scope.set((kind, settings, metadata))
    try:
        yield
    finally:
        scope.reset(token)


def log_payload(payload):
    """由兼容SDK在最终序列化后调用，记录真正即将发送的可见上下文。"""
    current = scope.get()
    if current is None:
        # 意图/摘要等没有开启model_ctx作用域的调用，不误标成主力Agent调用。
        return
    kind, settings, metadata = current
    messages = [
        {key: value for key, value in message.items() if key != 'reasoning_content'}
        for message in payload.get('messages', [])
    ]
    reasoning_tokens = sum(
        text_tokens(message.get('reasoning_content', ''), settings)
        for message in payload.get('messages', [])
    )
    # 序列化后的消息保留name；current_question前扣掉system，得到历史滑窗消息条数。
    # total_message_count另含system、当前问题、bundle及本轮工具链，不能与window_count混用。
    # 在SDK最终序列化后记录实际正文和工具定义；不接触HTTP客户端/配置对象。
    event(
        kind,
        **metadata,
        messages=messages,
        window_count=next(
            (
                index - 1
                for index, message in enumerate(messages)
                if message.get('name') == 'current_question'
            ),
            len(messages),
        ),
        total_message_count=len(messages),
        estimated_tokens=text_tokens(json.dumps(messages, ensure_ascii=False), settings)
        + reasoning_tokens,
        reasoning_tokens=reasoning_tokens,
        tools=payload.get('tools', []),
    )
