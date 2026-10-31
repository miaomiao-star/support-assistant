"""五个业务工具。订单/物流为固定演示，商品价格库存仍随机，不代表真实业务。"""

import random

from app.tools.demo_orders import get_order, get_logistics

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
        """按订单号查询订单状态和商品条件；返回固定演示数据。"""
        # Ch06卡片和查询共用固定源；未知编号不凭空生成。金额使用整数分。
        try:
            return get_order(order_id)
        except LookupError:
            return dict(ok=False, error='order_not_found')

    @tool(args_schema=ProductInput)
    async def query_product(keyword: str) -> dict:
        """查询商品价格和库存；返回随机演示数据。"""
        # 商品名称取输入关键词，价格/库存随机；再次查询可能不同，不能视为真实库存。
        return dict(
            mock=True,
            name=keyword,
            price_cents=random.randint(1000, 20000),
            stock=random.randint(0, 100),
            currency='CNY',
        )

    @tool(args_schema=OrderInput)
    async def query_logistics(order_id: str) -> dict:
        """按订单号直接查询物流，无需先查订单；返回固定演示数据。"""
        # 物流与卡片的签收状态一致，仍未接真实物流服务。
        try:
            return get_logistics(order_id)
        except LookupError:
            return dict(ok=False, error='order_not_found')

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
            context.conversation_id,
            description,
            ticket_type,
            context.ticket_no,
            context.call_id,
        )

    # 返回工具对象而非执行结果；此时还没有查询数据库或创建工单。
    return [query_order, query_product, query_logistics, query_faq, create_ticket]
