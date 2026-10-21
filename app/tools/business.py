"""五个业务工具。mock 数据每次随机生成，不代表真实业务状态。"""
import random

from langchain.tools import tool

from app.tools.schemas import FAQInput, OrderInput, ProductInput, TicketInput


def build_tools(context):
    """
    每次请求独立闭包：模型不能指定别人的会话或直接访问 Session。
    context 实际是本次请求的 ToolRegistry，其中持有 repo、会话 ID 等。
    内层函数可引用外层 context，这种写法叫“闭包”，避免使用全局会话变量。
    @tool 把函数包装成 LangChain 工具，提供 name、args_schema、ainvoke。
    注意：工具函数的三引号 docstring 会成为模型可见的工具描述。
    因此本次学习说明使用 # 注释，保持原 docstring/Field 描述不变。
    """
    @tool(args_schema=OrderInput)
    async def query_order(order_id: str) -> dict:
        """按订单号查询订单状态；返回随机演示数据。"""
        # 原样返回查询的订单号；状态与金额每次随机，不读取 orders 表或真实 API。
        # *_cents 使用整数分，避免用浮点数表达金额；mock=True 提醒上层是模拟。
        return dict(mock=True, order_id=order_id,
                    status=random.choice(['待付款', '已付款', '已发货', '已完成']),
                    total_cents=random.randint(1000, 50000), currency='CNY')

    @tool(args_schema=ProductInput)
    async def query_product(keyword: str) -> dict:
        """查询商品价格和库存；返回随机演示数据。"""
        # 商品名称取输入关键词，价格/库存随机；再次查询可能不同，不能视为真实库存。
        return dict(mock=True, name=keyword, price_cents=random.randint(1000, 20000),
                    stock=random.randint(0, 100), currency='CNY')

    @tool(args_schema=OrderInput)
    async def query_logistics(order_id: str) -> dict:
        """按订单号直接查询物流，无需先查订单；返回随机演示数据。"""
        # 直接接受订单号，确保“查物流”只需一个工具，符合本章单轮执行范围。
        # 三个字段分别随机，未模拟完整物流状态机或真实轨迹一致性。
        return dict(mock=True, order_id=order_id,
                    carrier=random.choice(['顺丰', '中通', '圆通']),
                    status=random.choice(['已揽收', '运输中', '派送中', '已签收']),
                    location=random.choice(['上海转运中心', '杭州转运中心', '收件地营业部']))

    @tool(args_schema=FAQInput)
    async def query_faq(keyword: str) -> dict:
        """用用户原话语义检索知识库中的政策、FAQ和使用说明；空列表即未命中。"""
        # 数据访问细节留在 repository，本工具只转交已校验的业务参数。
        # ok=True 表示查询执行成功；data=[] 表示没有匹配记录，两者不矛盾。
        return dict(ok=True, data=await context.repo.query_faq(keyword))

    @tool(args_schema=TicketInput)
    async def create_ticket(description: str, ticket_type: str) -> dict:
        """用户明确要求人工处理或建工单时创建工单，不执行退款。"""
        # call_id 关联工具申请，ticket_no 标识业务工单；二者职责不同。
        # 仓库负责在同一事务中保存工单和成功工具结果，本工具不自己 commit。
        return await context.repo.create_ticket(
            context.conversation_id, description, ticket_type,
            context.ticket_no, context.call_id)

    # 返回工具对象而非执行结果；此时还没有查询数据库或创建工单。
    return [query_order, query_product, query_logistics, query_faq, create_ticket]
