"""官方多Server客户端；远端元数据用于发现，不用于授予权限。"""

import asyncio
import json
import logging
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.tools.catalog import ToolSpec

logger = logging.getLogger(__name__)


def decode_result(result):
    """适配器ToolMessage优先读取结构化artifact，明确区分MCP执行错误。"""
    if isinstance(result, ToolMessage):
        # isError经官方adapter映射成status=error，先判失败，不能把错误文本当业务数据。
        if result.status == 'error':
            return dict(
                ok=False,
                category='fault',
                error='mcp_execution_failed',
                message='MCP工具报告执行失败。',
            )
        artifact = result.artifact or {}
        if artifact.get('structured_content') is not None:
            return artifact['structured_content']
        result = result.content
    if isinstance(result, list):
        # 文本回退只接受一个可解析JSON结果；多段文本不能凭拼接猜出业务事实。
        texts = [
            part['text']
            for part in result
            if isinstance(part, dict) and part.get('type') == 'text'
        ]
        if len(texts) != 1:
            raise ValueError('MCP返回不是单一结构化业务结果')
        result = texts[0]
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, dict):
        return result
    raise ValueError('MCP返回无法解析')


def business_fields(name, result):
    """三个自建业务工具只保留客服答复需要的字段；通用新工具不猜测语义。"""
    fields = {
        'query_logistics': [
            'mock',
            'order_id',
            'carrier',
            'tracking_no',
            'status',
            'events',
        ],
        'query_warranty': [
            'mock',
            'order_id',
            'in_warranty',
            'remaining_days',
            'status',
        ],
        'query_return_progress': [
            'mock',
            'order_id',
            'return_no',
            'status',
            'updated_at',
        ],
    }
    if not isinstance(result, dict) or result.get('ok') is False:
        return result
    if name not in fields:
        # 热新增工具无需在核心代码补名称分支；未知业务保留原合法JSON语义。
        return result
    selected = {key: result[key] for key in fields[name] if key in result}
    codes = {
        'IN_TRANSIT': '运输中',
        'DELIVERED': '已签收',
        'RECEIVED': '售后仓已收件',
        'REVIEWING': '退货审核中',
        'RETURNING': '退货运输中',
        'VALID': '在保',
        'EXPIRED': '已过保',
    }
    if selected.get('status') in codes:
        selected['status'] = codes[selected['status']]
    return selected


class MCPDiscovery:
    def __init__(self, catalog, connections, *, client=None, timeout=5):
        self.catalog, self.connections = catalog, connections
        self.client = client or MultiServerMCPClient(
            connections, handle_tool_errors=True
        )
        self.timeout = timeout
        self.errors = {}
        self.lock = asyncio.Lock()

    async def refresh(self):
        """每执行段重新list_tools；分Server隔离失败，更新目录时不留半份清单。"""
        async with self.lock:

            async def discover(server):
                try:
                    async with asyncio.timeout(self.timeout):
                        tools = await self.client.get_tools(server_name=server)
                    specs = []
                    for tool in tools:
                        schema = tool.args_schema
                        if not isinstance(schema, dict):
                            schema = schema.model_json_schema()

                        async def invoke(args, context, item=tool):
                            # 默认参数捕获本轮tool，避免循环闭包最终全指向最后一个工具。
                            # 传完整tool_call以保留artifact及status；仅传args会丢结构化产物。
                            response = await item.ainvoke(
                                dict(
                                    name=item.name,
                                    args=args,
                                    id=context.call_id,
                                    type='tool_call',
                                )
                            )
                            return business_fields(item.name, decode_result(response))

                        specs.append(
                            ToolSpec(
                                tool.name,
                                tool.description,
                                schema,
                                invoke,
                                source='mcp',
                                # 来源取本地连接键，不接受Server自报身份或只读声明来授权。
                                server=server,
                            )
                        )
                    return server, specs, None
                except Exception as exc:
                    logger.warning(
                        'MCP发现失败 server=%s type=%s', server, type(exc).__name__
                    )
                    return server, [], 'MCP服务暂时不可用'

            results = await asyncio.gather(
                *(discover(server) for server in self.connections)
            )
            self.errors = {}
            for server, specs, error in results:
                # 发现按Server并行，目录按来源整体替换；一台失败不移除其他来源。
                try:
                    self.catalog.replace_server(server, specs)
                except ValueError:
                    # 重名或非法Schema拒绝整份新清单，并下线该来源旧工具，防止用旧数据冒充成功。
                    self.catalog.replace_server(server, [])
                    error = 'MCP工具重名或Schema冲突，未登记'
                if error:
                    self.errors[server] = error
