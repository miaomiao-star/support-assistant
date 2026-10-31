"""独立建单确认接口；没有确认字段或没有收到建议时不能创建工单。"""

from typing import Literal

from pydantic import Field

from app.tools.schemas import TicketInput


class TicketConfirmation(TicketInput):
    # 继承description/ticket_type及原有校验；本章不另造一套工单业务字段。
    session_id: str = Field(min_length=1, max_length=128)
    # 这里只校验ID形式；建议是否存在、是否属于本会话仍由服务端checkpoint确认。
    offer_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    # 必须显式提交true；缺字段或false均不能当作用户已经确认。
    confirmed: Literal[True]
