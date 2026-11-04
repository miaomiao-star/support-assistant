"""售后MCP独立进程；随机演示在保/退货进度，不建业务表。"""

import argparse
import random
from datetime import datetime
from typing import Annotated
from pydantic import Field
from mcp.server.fastmcp import FastMCP


def build_server(port=8102):
    # 独立Streamable HTTP Server；无持久会话业务状态，不依赖客服进程或业务表。
    server = FastMCP(
        'MewHelp售后演示',
        host='127.0.0.1',
        port=port,
        stateless_http=True,
        json_response=True,
    )

    # 类型/Field约束导出给Client做统一校验；docstring是工具契约，不改作学习注释。
    @server.tool()
    def query_warranty(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按订单号查询模拟在保状态、剩余保修天数；不决定真实售后权益。"""
        # 是否在保与剩余天数来自同一个随机值，避免同一次响应内部自相矛盾。
        days = random.randint(-100, 365)
        return dict(
            mock=True,
            order_id=order_id,
            in_warranty=days > 0,
            remaining_days=max(0, days),
            status='VALID' if days > 0 else 'EXPIRED',
            internal_policy_id=12,
        )

    @server.tool()
    def query_return_progress(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按订单号查询模拟退货进度，返回退货单号与当前状态，不发起退货或退款。"""
        # 这只是查询模拟状态；政策资格仍由客服检索证据判断，不在这里发起业务写入。
        return dict(
            mock=True,
            order_id=order_id,
            return_no='MOCK-R' + str(random.randint(10000, 99999)),
            status=random.choice(['REVIEWING', 'RETURNING', 'RECEIVED']),
            updated_at=datetime.now().isoformat(timespec='minutes'),
            warehouse_code='WH001',
        )

    return server


if __name__ == '__main__':
    # 与物流端口分开，允许只重启其中一台；客服下一执行段重新发现其清单。
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8102)
    build_server(parser.parse_args().port).run(transport='streamable-http')
