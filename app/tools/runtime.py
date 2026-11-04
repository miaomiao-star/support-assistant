"""应用共享工具目录与每执行段独立上下文之间的桥。"""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from app.tools.catalog import Catalog, ToolSpec, LocalPolicy
from app.tools.business import build_tools
from app.tools.engine import ToolEngine, ExecutionContext
from app.tools.audit import AuditStore
from app.tools.mcp import MCPDiscovery

ROOT = Path(__file__).resolve().parents[2]


class ToolRuntime:
    # 生命周期与应用相同：共享目录/发现客户端，不共享某次聊天的repo和批准信息。
    def __init__(self, *, connections=None, policy_path=None, plugins_path=None):
        self.catalog = Catalog()
        self.policy_path = Path(policy_path or ROOT / 'config/tool-policy.json')
        self.plugins_path = Path(plugins_path or ROOT / 'tool_plugins')
        self.loaded_plugins = {}
        self.lock = asyncio.Lock()
        for item in build_tools(SimpleNamespace()):
            # 启动时只取元数据，SimpleNamespace不会执行查询或写入。
            if item.name == 'query_logistics':
                continue  # 正式清单只有MCP物流；旧ToolRegistry保留历史factory行为。

            async def invoke(args, context, name=item.name):
                # 调用时才把本请求上下文绑定给旧工具，避免不同会话互用repo/工单号。
                bound = next(tool for tool in build_tools(context) if tool.name == name)
                return await bound.ainvoke(args)

            self.catalog.register(
                ToolSpec(
                    item.name,
                    item.description,
                    item.args_schema.model_json_schema(),
                    invoke,
                    access='write' if item.name == 'create_ticket' else 'read',
                )
            )
        self.discovery = MCPDiscovery(self.catalog, connections or {})

    async def refresh(self):
        async with self.lock:
            # 将本地加载与MCP发现串成一次更新，防止并发请求登记两次同名工具。
            # 本地插件是开发者代码，与修改应用文件相同信任级，绝不加载远端代码。
            for path in sorted(self.plugins_path.glob('*.py')):
                if path.name.startswith('_'):
                    continue
                stamp = path.stat().st_mtime_ns
                if path in self.loaded_plugins:
                    # 只自动加载新文件；stamp是加载时记录，本章不据此热替换同名文件。
                    continue
                module_spec = importlib.util.spec_from_file_location(
                    'mewhelp_tool_' + path.stem, path
                )
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                module.register(self.catalog)
                self.loaded_plugins[path] = stamp
            await self.discovery.refresh()

    async def registry(self, repo, cid, raw, settings):
        # 普通聊天和暂停恢复都算新执行段：先刷新，再冻结工具Schema和权限快照。
        await self.refresh()
        registry = UnifiedRegistry(
            self.catalog.snapshot(),
            LocalPolicy.load(self.policy_path),
            repo,
            cid,
            raw,
            settings,
            errors=dict(self.discovery.errors),
        )
        registry.policy_path = self.policy_path
        return registry


class UnifiedRegistry:
    # 每个执行段独立创建，兼容旧Agent的registry接口；快照期间工具定义不变化。
    unified_tools = True

    def __init__(self, specs, policy, repo, cid, raw, settings, *, errors=None):
        self.specs, self.policy = specs, policy
        self.repo, self.conversation_id, self.user_text = repo, cid, raw
        self.settings, self.errors = settings, errors or {}
        self.request_key = ''
        self.ticket_requested = False
        self.tools = {name: spec.model_schema() for name, spec in specs.items()}
        self.read_only = {name: spec.access == 'read' for name, spec in specs.items()}

    def model_tools(self):
        return list(self.tools.values())

    def engine(self):
        # 每次调用创建独立ExecutionContext；目录中的handler只接受当前这份上下文。
        return ToolEngine(
            self.specs,
            self.policy,
            ExecutionContext(
                self.conversation_id, self.repo, self.request_key, self.ticket_requested
            ),
            audit=AuditStore(self.repo.db).write,
            timeout=self.settings.tool_timeout_seconds,
        )

    async def execute(self, call, **kwargs):
        # 权限撤销立即生效；新授权在下执行段生效，当前Schema不在运行中改写。
        current = LocalPolicy.load(
            getattr(self, 'policy_path', ROOT / 'config/tool-policy.json')
        )
        effective = LocalPolicy(
            # 取段初授权与当前授权的交集：撤销立即生效，新增授权留到下一执行段。
            {
                server: {
                    name: 'read'
                    for name in rules
                    if current.access(server, name) == 'read'
                }
                for server, rules in self.policy.mcp.items()
            }
        )
        engine = self.engine()
        engine.policy = effective
        return await engine.execute(call, **kwargs)
