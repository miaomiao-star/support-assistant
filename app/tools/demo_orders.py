"""可重复的演示订单；卡片、查单和物流共用，不模拟真实登录或真实订单服务。"""

from copy import deepcopy

_ORDERS = [
    dict(
        mock=True,
        order_id="1001",
        product_name="MH-W40饮水机",
        model="MH-W40",
        product_category="饮水机",
        status="已完成",
        total_cents=19900,
        currency="CNY",
        days_since_delivery=2,
        product_type="普通电子商品",
        intact=True,
        opened=False,
    ),
    dict(
        mock=True,
        order_id="1002",
        product_name="除臭猫砂",
        model=None,
        product_category="猫砂",
        status="已完成",
        total_cents=5900,
        currency="CNY",
        days_since_delivery=10,
        product_type="消耗品",
        intact=True,
        opened=True,
    ),
    dict(
        mock=True,
        order_id="1003",
        product_name="定制姓名猫牌",
        model=None,
        product_category="宠物配件",
        status="已完成",
        total_cents=3900,
        currency="CNY",
        days_since_delivery=1,
        product_type="定制类",
        intact=True,
        opened=False,
    ),
    dict(
        mock=True,
        order_id="1004",
        product_name="MH-F20喂食器",
        model="MH-F20",
        product_category="喂食器",
        status="已完成",
        total_cents=29900,
        currency="CNY",
        days_since_delivery=3,
        product_type="普通电子商品",
        intact=None,
        opened=None,
    ),
]


def list_orders():
    """返回副本，防止某个请求修改共享演示源。"""
    return deepcopy(_ORDERS)


def get_order(order_id):
    """未知编号明确不存在，不能按模型编造的编号生成订单。"""
    for order in _ORDERS:
        if order["order_id"] == order_id:
            return deepcopy(order)
    raise LookupError("未找到演示订单")


def get_logistics(order_id):
    """查同一个已存在订单，返回与签收状态一致的固定演示物流。"""
    order = get_order(order_id)
    return dict(
        mock=True,
        order_id=order_id,
        carrier="顺丰",
        status="已签收",
        location="收件地营业部",
        days_since_delivery=order["days_since_delivery"],
    )
