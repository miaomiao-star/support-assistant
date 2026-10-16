from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractRequest(BaseModel):
    """去除首尾空白后仍需有文本；无效输入由 FastAPI 返回 422。"""
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(min_length=1)


class AfterSalesTicket(BaseModel):
    """仅从用户原文提取的售后诉求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    order_id: str | None = Field(default=None, min_length=1, description="原文明示的订单号；缺失为 null")
    # None 序列化为 JSON null；字符串 "null" 与 None 是两种不同值。
    request_type: Literal["退款", "换货", "维修", "投诉", "其他"]
    # Literal 是封闭取值集合，模型输出其他类型会解析失败，而非任意放行。
    expected_solution: str = Field(min_length=1, description="一句话概括期望方案；不明确时说明需进一步确认")
