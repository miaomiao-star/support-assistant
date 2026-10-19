"""工具白名单、Schema 校验、有限重试及安全错误结果。"""
import asyncio
import hashlib
import logging
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.tools.business import build_tools

logger = logging.getLogger(__name__)


class ToolRegistry:
    # 每条聊天请求创建一个实例；不是所有用户共用同一个可变的执行上下文。
    def __init__(self, repo, conversation_id, user_text, timeout=5):
        self.repo, self.conversation_id, self.user_text = repo, conversation_id, user_text
        self.timeout = timeout
        # tools：名称 → 工具对象；read_only：名称 → 是否允许只读重试。
        # 模型不能靠随意生成一个名字就执行系统函数，只能命中这个白名单。
        self.tools, self.read_only = {}, {}
        self.call_id = self.ticket_no = ''
        # 区分不同聊天请求。用户重发同一句话会生成新值，本章不做跨请求去重。
        self.request_key = uuid4().hex
        for item in build_tools(self):
            # 只有 create_ticket 会写业务状态；其余四个按只读操作处理。
            self.register(item, read_only=item.name != 'create_ticket')

    def register(self, item, read_only=True):
        """
        注册时拒绝重复名称，避免一个工具被另一个同名工具意外覆盖。
        """
        if item.name in self.tools:
            raise ValueError('Duplicate tool name')
        self.tools[item.name] = item
        self.read_only[item.name] = read_only

    async def execute(self, call):
        """
        call 来自模型的工具申请：{name, args, id, type}。
        “选择了工具”与“获准执行工具”是两步，下面先逐项检查再执行。
        """
        name = call['name']
        if name not in self.tools:
            return dict(ok=False, error='unknown_tool')
        item = self.tools[name]
        try:
            # model_validate 按工具自己的 Schema 校验/标准化输入；
            # model_dump 转回字典，交给工具调用器按参数名传给业务函数。
            args = item.args_schema.model_validate(call['args']).model_dump()
        except (ValidationError, TypeError):
            # 不将校验异常原文发给模型/浏览器，异常可能带有原始输入。
            return dict(ok=False, error='invalid_arguments')
        if name == 'query_faq' and args['keyword'] not in self.user_text:
            # 防止模型把“邮费”改成“运费”，否则会掩盖 LIKE 的同义词漏召回。
            return dict(ok=False, error='keyword_not_literal')
        if name == 'create_ticket' and not any(w in self.user_text for w in ('人工', '工单')):
            # 当前仅是演示级关键词门槛，并非完整的语义确认、登录或权限校验。
            # 不能把这个分支理解成生产系统已经可靠识别了所有肯定/否定意图。
            return dict(ok=False, error='confirmation_required')
        self.call_id = call['id']
        # 同一次工具调用的工单号固定；写入未知结果时也不会盲重试。
        # 上游调用 ID 可能跨回合重复，必须同时纳入本次请求标识。
        key = f'{self.conversation_id}:{self.request_key}:{self.call_id}'.encode()
        self.ticket_no = 'T' + hashlib.sha256(key).hexdigest()[:30]
        # attempts 是“总尝试次数”：2 = 首次执行 + 最多重试 1 次。
        # 写操作遇到超时可能已经提交，盲重试会有重复副作用，所以只尝试一次。
        attempts = 2 if self.read_only[name] else 1
        for attempt in range(attempts):
            try:
                # asyncio.timeout 限制这一次工具执行的等待时间，不限制整轮模型耗时。
                # 协程收到取消后需要配合清理；它不是能撤销已提交业务的回滚按钮。
                async with asyncio.timeout(self.timeout):
                    result = await item.ainvoke(args)
                # FAQ/工单已经返回标准 {ok,...}；三个 mock 返回业务数据，
                # 在这里统一包装为 {ok:True,data:...}，便于服务层读取状态。
                if isinstance(result, dict) and 'ok' in result:
                    return result
                return dict(ok=True, data=result)
            except (TimeoutError, ConnectionError, OperationalError) as exc:
                # 只有超时/临时网络故障可重试；认证、SQL 等错误不重试。
                # 1205/1213 是锁等待超时/死锁，2006/2013 是连接丢失。
                # 这是示例的有限白名单，不是对所有 OperationalError 一概重试。
                transient = not isinstance(exc, OperationalError) or (
                    getattr(exc.orig, 'args', [None])[0] in (1205, 1213, 2006, 2013))
                if transient and attempt + 1 < attempts:
                    # 短暂让出事件循环后重试，避免失败后立刻连续冲击数据源。
                    await asyncio.sleep(.05)
                    continue
                return dict(ok=False, error='timeout' if isinstance(exc, TimeoutError) else 'unavailable')
            except Exception as exc:
                # 日志只记工具名和异常类型，不回显数据库连接串、参数或上游原文。
                # asyncio.CancelledError 不属于普通 Exception，会继续向外传播，
                # 这样浏览器断开时工具不会被误当成普通失败后继续执行。
                logger.warning('Tool %s failed: %s', name, type(exc).__name__)
                return dict(ok=False, error='execution_failed')
        # 每次循环都会 return/raise 或进入下一次尝试；正常流程不会到这里。
        raise AssertionError('Unreachable')
