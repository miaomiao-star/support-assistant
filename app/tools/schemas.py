"""模型能看见的业务参数；会话、数据库连接由后端注入。"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolInput(BaseModel):
    # 所有工具参数共用规则：拒绝未声明字段，去掉字符串首尾空白。
    # 例如模型多传 sql 或 conversation_id，不会被悄悄接受。
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class OrderInput(ToolInput):
    # 字符串可保留订单号前导零；pattern 限定字符，不查询订单是否真的存在。
    order_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$', description='用户提供的订单号')


class ProductInput(ToolInput):
    # 这里只限制输入形式；本章商品工具仍返回随机模拟数据。
    keyword: str = Field(min_length=1, max_length=64, description='商品名称或编号')


class FAQInput(ToolInput):
    # Field(description=...) 会进入提供给模型的工具参数说明。
    # “必须来自原话”还要在 registry 中检查，不能只依赖这段描述。
    keyword: str = Field(min_length=1, max_length=64, description='用户原话中的连续关键词，禁止同义改写')


class TicketInput(ToolInput):
    # 只让模型提供业务描述和类型，实际会话 ID/工单号由后端注入。
    description: str = Field(min_length=1, max_length=2000, description='用户问题描述')
    # Literal 限定可选值，模型传“退款”等未列出的类型会校验失败。
    ticket_type: Literal['售后', '投诉', '咨询']
