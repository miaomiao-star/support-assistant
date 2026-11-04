"""物流MCP独立进程；随机模拟轨迹，不连接真实物流或数据库。"""

import argparse
import random
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Annotated


def build_server(port=8101):
    # SDK1.x的host/port放构造器，run只选传输；此进程独立于客服Web服务。
    server = FastMCP(
        'MewHelp物流演示',
        host='127.0.0.1',
        port=port,
        stateless_http=True,
        json_response=True,
    )

    # SDK从类型/Field生成参数Schema，从docstring生成模型可见描述；下方描述保持原样。
    @server.tool()
    def query_logistics(
        order_id: Annotated[str, Field(min_length=1, max_length=64)],
    ) -> dict:
        """按用户给出的订单号查询模拟物流轨迹，含承运商、物流状态和时间地点。"""
        # 每次现造随机mock，不和固定演示订单共享状态，也不查询真实物流或数据库。
        delivered = random.choice([True, False])
        now = datetime.now()
        return dict(
            mock=True,
            order_id=order_id,
            carrier=random.choice(['顺丰', '中通', '圆通']),
            tracking_no='MOCK' + str(random.randint(10000000, 99999999)),
            status='DELIVERED' if delivered else 'IN_TRANSIT',
            events=[
                dict(
                    time=(now - timedelta(hours=12)).isoformat(timespec='minutes'),
                    location='始发分拨中心',
                    detail='包裹已揽收',
                ),
                dict(
                    time=now.isoformat(timespec='minutes'),
                    location='收件地网点',
                    detail='已签收' if delivered else '派送中',
                ),
            ],
            # 客服侧按本地适配规则剔除该内部字段；Server原始返回与回答投影分开。
            internal_route_id=random.randint(1, 999),
        )

    # Ch08热新增验收：只新增Server工具，客服核心图无需认识这个名称。
    @server.tool()
    def query_carrier_hotline() -> dict:
        """查询物流承运商的模拟客服电话；只是演示号码，不可作为真实客服联系方式。"""
        return dict(mock=True, carrier='模拟承运商', hotline='演示热线400-000-0008')

    # 最终热新增验收在客服进程启动后才投放此工具，并热更本地逐工具read规则。
    @server.tool()
    def query_pickup_hours() -> dict:
        """查询物流网点的模拟营业时间，不查询物流轨迹；返回演示营业时段。"""
        return dict(mock=True, pickup_hours='09:00-18:00', site='模拟物流网点')

    return server


if __name__ == '__main__':
    # 只有作为命令运行才监听；测试可构建Server检查工具而不启动端口。
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8101)
    server = build_server(parser.parse_args().port)
    server.run(transport='streamable-http')
