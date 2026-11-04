"""工单批准只有服务端凭据和布尔选择，前端不可重新指定参数。"""

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class TicketDecision(BaseModel):
    # 不接受description/ticket_type/approved等额外字段；参数只取PG保存的申请。
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=128)
    preview_id: str = Field(min_length=1, max_length=160)
    # 严格布尔，拒绝把字符串"false"或数字隐式转换成确认选择。
    confirmed: StrictBool
