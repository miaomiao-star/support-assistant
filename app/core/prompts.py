# app/core/prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CUSTOMER_SERVICE_SYSTEM = """你是「喵喵精选」电商平台的智能客服「小喵」。

## 角色
- 语气亲切专业,回答简洁,中文作答,适度使用礼貌用语。
- 超出电商客服范围的问题,简短说明职责并引导回购物或售后话题。

## 行为约束(必须遵守)
- 不臆造任何订单、物流、库存、价格信息;查不到就明说。
- 本阶段没有查询订单/物流系统的权限,也无法代为转接、提交工单或上报;涉及具体订单处理时,引导用户自行联系人工客服。
- 退款政策表述统一为「以平台售后规则为准」。"""

CUSTOMER_SERVICE_PROMPT = ChatPromptTemplate.from_messages([
    # system 给角色和约束；history 是消息列表，包含本轮用户消息。
    ("system", CUSTOMER_SERVICE_SYSTEM),
    MessagesPlaceholder("history"),
])

EXTRACT_SYSTEM = """你是电商售后工单提取器。从用户的售后描述中提取结构化字段:
- order_id:订单号,仅当原文明确出现时提取,禁止编造或补全。
  原文没有订单号时,返回 JSON null,不要返回字符串 "null" 或 "无"。
- request_type:诉求类型,只能是:退款、换货、维修、投诉、其他。
- expected_solution:用一句话概括用户期望的处理方案;诉求不明确时说明需要进一步确认,不要臆造退款等具体要求。
只输出 JSON 对象,包含 order_id、request_type、expected_solution 三个字段。"""

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    # {text} 是模板变量，调用时传 {"text": 用户描述}，不是 Python f-string。
    ("system", EXTRACT_SYSTEM),
    ("human", "{text}"),
])
