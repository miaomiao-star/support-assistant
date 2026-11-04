"""应用共享的工具目录；这里只保存元数据和无会话的执行适配器。"""

import json
import logging
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    # 目录条目只绑定无会话的适配器；repo、会话号和批准信息到执行时再传。
    name: str
    description: str
    schema: dict
    handler: object
    source: str = 'builtin'
    server: str | None = None
    access: str = 'read'

    def model_schema(self):
        """交给bind_tools的统一OpenAI兼容Schema，不混入内部权限参数。"""
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': deepcopy(self.schema),
            },
        }


class Catalog:
    def __init__(self):
        self.entries = {}
        # 版本只描述目录更新，不是数据库版本，也不代表某个会话的执行进度。
        self.version = 0

    def register(self, spec):
        """拒绝重名和非法Schema，防止外部工具替换本地同名实现。"""
        if spec.name in self.entries:
            raise ValueError('工具名称重复：' + spec.name)
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', spec.name) or not spec.description:
            raise ValueError('工具名称或描述不合法')
        try:
            Draft202012Validator.check_schema(spec.schema)

            # 允许内嵌$defs，拒绝网络引用。校验不能按远端Schema去请求任意地址。
            def refs(value):
                if isinstance(value, dict):
                    if '$ref' in value and not value['$ref'].startswith('#'):
                        raise ValueError('只允许Schema内部引用')
                    for child in value.values():
                        refs(child)
                elif isinstance(value, list):
                    for child in value:
                        refs(child)

            refs(spec.schema)
        except Exception as exc:
            raise ValueError('工具参数Schema不合法') from exc
        # 拷贝可变Schema，避免注册方随后修改原字典而改变已登记工具的参数契约。
        self.entries[spec.name] = ToolSpec(
            spec.name,
            spec.description,
            deepcopy(spec.schema),
            spec.handler,
            spec.source,
            spec.server,
            spec.access,
        )
        self.version += 1

    def snapshot(self):
        """每次执行段持有独立Schema副本；共享目录随后更新不改正在跑的调用。"""
        return {
            name: ToolSpec(
                item.name,
                item.description,
                deepcopy(item.schema),
                item.handler,
                item.source,
                item.server,
                item.access,
            )
            for name, item in self.entries.items()
        }

    def replace_server(self, server, specs):
        """先在临时目录完整校验，成功后整体替换该Server，禁止部分覆盖。"""
        candidate = Catalog()
        # 保留其他来源；该Server已删除的工具不会被带入新目录，故障时也可整体下线。
        for item in self.entries.values():
            if item.server != server:
                candidate.register(item)
        for item in specs:
            candidate.register(item)
        self.entries = candidate.entries
        self.version += 1


class LocalPolicy:
    # 只接受本地逐工具规则；Server的description/annotations不进入权限判定。
    def __init__(self, mcp=None):
        self.mcp = deepcopy(mcp or {})

    @classmethod
    def load(cls, path):
        """每执行段重读本地文件；坏配置关闭外部权限，不沿用旧授权。"""
        try:
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            rules = data['mcp']
            if not isinstance(rules, dict) or any(
                not isinstance(v, dict)
                or any(access != 'read' for access in v.values())
                for v in rules.values()
            ):
                raise ValueError('仅允许逐工具read规则')
            return cls(rules)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning('工具权限配置不可用，外部工具全部拒绝')
            return cls()

    def access(self, server, name):
        # 工具已被发现不等于已被授权；缺少任一层配置时默认拒绝。
        return self.mcp.get(server, {}).get(name, 'deny')
