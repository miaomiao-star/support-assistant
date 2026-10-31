"""演示退款申请：MySQL业务事实与图建议分开，提交成功不代表资金已退款。"""

import hashlib
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, ForeignKey, JSON, String, Text, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base, ID

REASONS = ["不喜欢/不需要", "质量问题", "运输破损", "发错货", "少发漏发", "其他"]


class OrderSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(min_length=1, max_length=128)
    selection_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    order_id: str = Field(min_length=1, max_length=64)


class RefundSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    session_id: str = Field(min_length=1, max_length=128)
    offer_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    reason: Literal[
        "不喜欢/不需要", "质量问题", "运输破损", "发错货", "少发漏发", "其他"
    ]
    description: str = Field(default="", max_length=2000)
    confirmed: Literal[True]


class RefundApplication(Base):
    __tablename__ = "refund_applications"
    application_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ID, ForeignKey("conversations.id"), index=True
    )
    offer_id: Mapped[str] = mapped_column(String(32))
    order_id: Mapped[str] = mapped_column(String(64))
    order_snapshot: Mapped[dict] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="待处理")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def receipt(row):
    return dict(
        ok=True,
        mock=True,
        application_no=row.application_no,
        order_id=row.order_id,
        status=row.status,
    )


class RefundStore:
    def __init__(self, database):
        self.database = database

    async def submit(self, cid, offer_id, order, reason, description):
        """稳定业务主键兜住MySQL已提交、Postgres回写或HTTP响应丢失的重试。"""
        key = "R" + hashlib.sha256(f"{cid}:{offer_id}".encode()).hexdigest()[:30]
        try:
            async with self.database.sessions.begin() as session:
                row = await session.get(RefundApplication, key)
                if row is None:
                    row = RefundApplication(
                        application_no=key,
                        conversation_id=cid,
                        offer_id=offer_id,
                        order_id=order["order_id"],
                        order_snapshot=order,
                        reason=reason,
                        description=description,
                        status="待处理",
                    )
                    session.add(row)
                    await session.flush()
                return receipt(row)
        except IntegrityError:
            # 唯一键冲突只认可同会话同建议；其他约束失败不能伪装成提交成功。
            async with self.database.sessions() as session:
                row = await session.get(RefundApplication, key)
                if (
                    row is None
                    or row.conversation_id != cid
                    or row.offer_id != offer_id
                ):
                    raise
                return receipt(row)
