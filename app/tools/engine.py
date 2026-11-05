"""唯一工具执行引擎：校验、授权、有限重试、结果与独立审计。"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter

import httpx
from jsonschema import Draft202012Validator
from referencing import Registry
from sqlalchemy.exc import OperationalError
from app.learning.observability import trace_tool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    # 每次执行独立创建；业务工具从这里拿repo和稳定编号，不能存回应用共享目录。
    conversation_id: int | None
    repo: object = None
    request_id: str = ''
    ticket_requested: bool = False
    call_id: str = ''
    ticket_no: str = ''


def fingerprint(context, call):
    # 批准同时绑定会话、请求、调用和完整参数；改描述或借用其他会话的确认都会失配。
    # sort_keys消除字典键顺序差异，哈希用于绑定，不是客户端可携带的授权令牌。
    return hashlib.sha256(
        json.dumps(
            [
                context.conversation_id,
                context.request_id,
                call['id'],
                call['name'],
                call['args'],
            ],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Approval:
    """只由已核验的确认接口/图节点创建，不接受模型JSON里的approved。"""

    digest: str

    @classmethod
    def for_call(cls, context, call):
        return cls(fingerprint(context, call))


def failure(category, message, *, error=None, details=None):
    result = dict(ok=False, category=category, error=error or category, message=message)
    if details:
        result['details'] = details
    return result


def normalize(result):
    """剥离协议包装，保留业务语义；业务空不是网络故障。"""
    if result is None or result == [] or result == {}:
        return failure('not_found', '查询没有找到匹配结果，请核对信息。')
    if isinstance(result, dict) and result.get('ok') is False:
        error = result.get('error', 'execution_failed')
        if error.endswith('not_found') or result.get('category') == 'not_found':
            return failure(
                'not_found', '没有找到对应业务记录，请核对编号。', error=error
            )
        return failure(
            result.get('category', 'fault'),
            result.get('message', '工具未能完成操作。'),
            error=error,
        )
    if isinstance(result, dict) and result.get('ok') is True:
        if result.get('data', 1) in (None, [], {}):
            return failure('not_found', '查询没有找到匹配结果，请核对信息。')
        return result
    return dict(ok=True, data=result)


def transport_error(exc):
    """anyio/MCP可把底层网络错误放进异常组；只在全部叶子都临时时重试。"""
    if isinstance(exc, BaseExceptionGroup):
        items = [transport_error(item) for item in exc.exceptions]
        return all(item[0] for item in items), all(item[0] for item in items) and any(
            item[1] for item in items
        )
    timeout = isinstance(exc, (TimeoutError, httpx.TimeoutException))
    transient = timeout or isinstance(exc, (ConnectionError, httpx.NetworkError))
    if isinstance(exc, OperationalError):
        # 只把已列明的锁等待/死锁/连接丢失识别为暂时故障，不重试所有SQL错误。
        transient = getattr(exc.orig, 'args', [None])[0] in (1205, 1213, 2006, 2013)
    return transient, timeout


class ToolEngine:
    def __init__(self, tools, policy, context, *, audit=None, timeout=5):
        self.tools, self.policy, self.context = tools, policy, context
        self.audit, self.timeout = audit, timeout

    def check(self, call, *, mode_allowed=True):
        """准备预览也复用校验，但不在此写审计，待调用最终结局才落一条。"""
        spec = self.tools.get(call['name'])
        if not spec:
            return failure('permission_denied', '工具未注册，禁止调用。')
        if not mode_allowed or (
            spec.source == 'mcp'
            and self.policy.access(spec.server, spec.name) != 'read'
        ):
            return failure('permission_denied', '本地权限规则不允许本次工具调用。')
        try:
            # 检查原始JSON值，不做Pydantic式类型转换；空Registry禁止解析远程引用。
            errors = sorted(
                Draft202012Validator(spec.schema, registry=Registry()).iter_errors(
                    call.get('args')
                ),
                key=lambda e: str(list(e.path)),
            )
            if errors:
                # 回灌字段路径和原因让模型纠参；限制条数避免错误说明挤占上下文。
                details = [
                    dict(
                        field='.'.join(map(str, e.absolute_path)) or '$',
                        reason=e.message,
                    )
                    for e in errors[:8]
                ]
                return failure(
                    'invalid_arguments',
                    '参数不合法，请按说明补充或修正后再调用。',
                    details=details,
                )
        except Exception:
            return failure('invalid_arguments', '参数Schema无法校验，请联系维护人员。')
        return None

    @trace_tool
    async def execute(
        self,
        call,
        *,
        approval=None,
        explicit_ticket=None,
        mode_allowed=True,
        denied_reason=None,
        argument_error=None,
    ):
        """内部重试算同一次逻辑调用。审计失败绝不改写已经发生的业务结局。"""
        started = perf_counter()
        spec = self.tools.get(call['name'])
        retry_count = 0
        result = self.check(call, mode_allowed=mode_allowed)
        status = (
            '校验拦下'
            if result and result['category'] == 'invalid_arguments'
            else '权限拒绝'
        )
        write = bool(spec and spec.access == 'write')
        requested = (
            self.context.ticket_requested
            if explicit_ticket is None
            else explicit_ticket
        )
        if denied_reason:
            # 取消预览等服务端决定必须胜过模型提交的参数，且仍走最后的统一审计。
            result = failure('permission_denied', denied_reason)
            status = '权限拒绝'
        if argument_error and (
            result is None or result['category'] == 'invalid_arguments'
        ):
            # 图层可补业务参数错误（例如换了订单号），但不能覆盖已有权限拒绝。
            result = argument_error
            status = '校验拦下'
        if (
            not result
            and write
            and (
                spec.source != 'builtin'
                or spec.name != 'create_ticket'
                or not requested
                or approval != Approval.for_call(self.context, call)
            )
        ):
            # 只有内置建单获准写入；即使外部工具伪造用途声明，也没有这条授权路径。
            result = failure(
                'permission_denied', '写操作必须由客户明确要求，并确认本次工单预览。'
            )
            status = '权限拒绝'
        cancelled = False
        if not result:
            self.context.call_id = call['id']
            key = (
                f'{self.context.conversation_id}:{self.context.request_id}:{call["id"]}'
            )
            self.context.ticket_no = 'T' + hashlib.sha256(key.encode()).hexdigest()[:30]
            # 同一逻辑调用生成同一工单号，供业务层防重；这不等于跨MySQL/PG精确一次。
            # range限制的是总尝试数：读最多2次（重试1次），写始终只尝试1次。
            for attempt in range(1 if write else 2):
                try:
                    async with asyncio.timeout(self.timeout):
                        raw = await spec.handler(call['args'], self.context)
                    result = normalize(raw)
                    # 完整业务结果也必须能按中文JSON回灌，不能把不可序列化对象算成功。
                    json.dumps(result, ensure_ascii=False, allow_nan=False)
                    status = '成功' if result['ok'] else '失败'
                    break
                except asyncio.CancelledError:
                    # 先尽力留下取消结局，末尾再向上传播取消；写入可能已经发生。
                    cancelled = True
                    result = failure(
                        'fault',
                        '工具调用被取消；写入结果需核对，不能断言未执行。',
                        error='cancelled',
                    )
                    status = '失败'
                    break
                except Exception as exc:
                    transient, timeout = transport_error(exc)
                    if transient and not write and attempt == 0:
                        # 只有抛出的暂时故障进入重试；正常返回的业务空/失败不会到这里。
                        retry_count += 1
                        await asyncio.sleep(0.05)
                        continue
                    status = '超时' if timeout else '失败'
                    message = '工具超时。' if timeout else '工具暂时无法完成操作。'
                    if write:
                        message += '写入结果未知，请核对工单记录；本次没有自动重试。'
                    result = failure(
                        'fault',
                        message,
                        error='timeout' if timeout else 'execution_failed',
                    )
                    logger.warning(
                        '工具执行失败 name=%s type=%s', call['name'], type(exc).__name__
                    )
                    break
        if self.audit:
            # 每次逻辑调用只汇总一条，包含实际重试次数；审计另开事务且最多等待2秒。
            # duration在写审计前取值，包含校验/执行/重试等待，不把审计等待算成工具耗时。
            try:
                async with asyncio.timeout(2):
                    await self.audit(
                        conversation_id=self.context.conversation_id,
                        tool_call_id=call['id'][:64],
                        tool_name=call['name'][:128],
                        tool_source=spec.source if spec else 'builtin',
                        mcp_server=spec.server if spec else None,
                        arguments=call.get('args'),
                        result_summary=json.dumps(result, ensure_ascii=False)[:4000],
                        status=status,
                        error_message=(
                            None
                            if result['ok']
                            else (
                                result['message']
                                + ' '
                                + json.dumps(
                                    result.get('details', []), ensure_ascii=False
                                )
                            )[:512]
                        ),
                        retry_count=retry_count,
                        duration_ms=max(0, int((perf_counter() - started) * 1000)),
                    )
            except Exception as exc:
                logger.warning(
                    '工具审计写入失败 name=%s type=%s', call['name'], type(exc).__name__
                )
        if cancelled:
            raise asyncio.CancelledError
        return result
