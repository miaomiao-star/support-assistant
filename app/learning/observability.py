"""本地Langfuse入口：图只绑定一次；请求上下文与意图不放进共享handler可变字段。"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import json
import logging
from functools import wraps
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
ACTIVE = ContextVar('mewhelp_trace', default=None)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='LANGFUSE_',
        env_file=Path(__file__).resolve().parents[2] / '.env.observability',
        extra='ignore',
    )
    enabled: bool = False
    base_url: str | None = None
    public_key: str | None = None
    secret_key: SecretStr | None = None


def mask(*, data, **kwargs):
    blocked = {
        'reasoning_content',
        'reasoning',
        'authorization',
        'api_key',
        'chat_api_key',
        'embed_api_key',
        'rerank_api_key',
        'password',
        'secret_key',
        'dsn',
        'database_url',
        'headers',
        'repo',
        'connection',
        'access_token',
        'refresh_token',
    }

    def clean(value, depth=0):
        if depth > 20:
            return '<嵌套过深>'
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, dict):
            return {
                str(k): (
                    '[已隐藏]'
                    if str(k).lower() in blocked
                    or str(k).lower().endswith(('_api_key', '_password', '_secret_key'))
                    else clean(v, depth + 1)
                )
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(v, depth + 1) for v in value]
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (dict, list)):
                    return clean(parsed, depth + 1)
            except (ValueError, TypeError):
                pass
            return value
        if value is None or isinstance(value, (int, float, bool)):
            return value
        return '<' + type(value).__name__ + '>'

    return clean(data)


class EmptyObservation:
    def update(self, **kwargs):
        return self


class Tracing:
    def __init__(self, settings=None, *, client=None, handler=None):
        self.settings = settings or ObservabilitySettings()
        self.client, self.handler = None, None
        if self.settings.enabled:
            url = urlsplit(self.settings.base_url or '')
            if (
                url.scheme not in ('http', 'https')
                or url.hostname not in ('127.0.0.1', 'localhost', '::1')
                or url.username
                or url.password
            ):
                raise ValueError('本章Langfuse必须显式配置本机地址，禁止默认云端')
            if not self.settings.public_key or not self.settings.secret_key:
                raise ValueError('缺少本地Langfuse项目凭据')
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            self.client = client or Langfuse(
                public_key=self.settings.public_key,
                secret_key=self.settings.secret_key.get_secret_value(),
                base_url=self.settings.base_url,
                mask=mask,
                timeout=30,  # 本机低内存部署的冷请求超过5秒；导出在后台等待，不阻塞工具执行。
            )
            self.handler = handler or CallbackHandler(
                public_key=self.settings.public_key
            )

    def bind(self, graph):
        if not self.handler:
            return graph
        bound = graph.with_config({'callbacks': [self.handler]})
        # 当前LangGraph重建配置对象时不保留自定义属性；显式延续应用的服务选择标记。
        for name in (
            'formal_router',
            'layered_context',
            'unified_tools',
            'learning_tools',
        ):
            if hasattr(graph, name):
                setattr(bound, name, getattr(graph, name))
        return bound

    @contextmanager
    def request(self, name, data, session):
        if not self.client:
            yield EmptyObservation()
            return
        existing = ACTIVE.get()
        if existing and existing['owner'] is self:
            # 新问题内部取消旧预览是同一个HTTP请求的子操作，不伪造第二个根意图。
            with observation(name, data, kind='chain') as child:
                yield child
            return
        from langfuse import propagate_attributes

        with self.client.start_as_current_observation(
            name=name,
            as_type='chain',
            input=data,
            metadata={'app_root': True, 'intent': '未分类'},
        ) as root:
            state = dict(owner=self, root=root, intent='未分类')
            token = ACTIVE.set(state)
            try:
                with propagate_attributes(
                    session_id=session,
                    metadata={'application': 'MewHelp', 'chapter': 'ch09'},
                ):
                    yield root
            finally:
                root.update(
                    metadata={
                        'app_root': True,
                        'intent': state['intent'],
                        'operation': state.get('operation'),
                    }
                )
                ACTIVE.reset(token)

    def intent(self, intent, operation):
        state = ACTIVE.get()
        if state and state['owner'] is self:
            state.update(intent=intent, operation=operation)
            state['root'].update(
                metadata={'app_root': True, 'intent': intent, 'operation': operation}
            )

    async def observations(self, trace_id=None):
        if not self.client:
            raise ValueError('Langfuse未启用')
        until = datetime.now(timezone.utc)
        params = {
            'fields': 'core,basic,usage,metadata',
            'limit': 1000,
            'fromStartTime': (until - timedelta(days=7)).isoformat(),
            'toStartTime': until.isoformat(),
        }
        if trace_id:
            params['traceId'] = trace_id
        result, cursors = [], set()
        async with httpx.AsyncClient(
            base_url=self.settings.base_url,
            auth=(
                self.settings.public_key,
                self.settings.secret_key.get_secret_value(),
            ),
            timeout=20,
            trust_env=False,
        ) as client:
            for _ in range(100):
                response = await client.get(
                    '/api/public/v2/observations', params=params
                )
                response.raise_for_status()
                data = response.json()
                result.extend(data['data'])
                cursor = data.get('meta', {}).get('cursor')
                if not cursor:
                    return result
                if cursor in cursors:
                    raise RuntimeError('Langfuse观察游标重复')
                cursors.add(cursor)
                params['cursor'] = cursor
        raise RuntimeError('观察结果过多，请收窄统计时间')

    async def close(self):
        if self.client:
            import asyncio

            await asyncio.to_thread(self.client.flush)


@contextmanager
def observation(name, data, *, kind='span'):
    state = ACTIVE.get()
    if not state or not state['owner'].client:
        yield EmptyObservation()
        return
    with state['owner'].client.start_as_current_observation(
        name=name, as_type=kind, input=data, metadata={'intent': state['intent']}
    ) as span:
        yield span


def aggregate_costs(observations):
    unique = {r['id']: r for r in observations}
    intents = {}
    for row in unique.values():
        metadata = row.get('metadata') or {}
        if metadata.get('intent') and (
            metadata.get('app_root') or row['traceId'] not in intents
        ):
            intents[row['traceId']] = metadata['intent']
    groups = {}
    for row in unique.values():
        if row.get('type', '').upper() != 'GENERATION':
            continue
        intent = intents.get(row['traceId'], '未分类')
        group = groups.setdefault(
            intent,
            dict(
                intent=intent,
                generations=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                missing_usage_generations=0,
                unpriced_generations=0,
                cost_usd=None,
            ),
        )
        group['generations'] += 1
        usage = row.get('usageDetails') or {}
        if not usage and not any(
            row.get(key) for key in ('inputUsage', 'outputUsage', 'totalUsage')
        ):
            group['missing_usage_generations'] += 1
        else:
            # v4的基础input/output不包含拆出的缓存/推理，用服务端合计字段避免漏算。
            input_tokens = row.get('inputUsage')
            if input_tokens is None:
                input_tokens = sum(
                    value
                    for key, value in usage.items()
                    if key == 'input' or key.startswith('input_')
                )
            output_tokens = row.get('outputUsage')
            if output_tokens is None:
                output_tokens = sum(
                    value
                    for key, value in usage.items()
                    if key == 'output' or key.startswith('output_')
                )
            total_tokens = row.get('totalUsage')
            if total_tokens is None:
                total_tokens = usage.get('total', input_tokens + output_tokens)
            group['input_tokens'] += input_tokens
            group['output_tokens'] += output_tokens
            group['total_tokens'] += total_tokens
        cost = row.get('totalCost')
        if cost is None:
            group['unpriced_generations'] += 1
        else:
            group['cost_usd'] = (group['cost_usd'] or 0) + cost
    return sorted(groups.values(), key=lambda r: r['total_tokens'], reverse=True)


def trace_tool(function):
    @wraps(function)
    async def wrapped(self, call, **kwargs):
        with observation(
            'tool:' + call['name'],
            dict(tool_call_id=call['id'], arguments=call.get('args')),
            kind='tool',
        ) as span:
            result = await function(self, call, **kwargs)
            span.update(output=result)
            return result

    return wrapped
