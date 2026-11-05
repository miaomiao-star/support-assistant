"""严格映射用户Ch09原DDL，真实迁移不用metadata.create_all。"""

from datetime import datetime
from sqlalchemy import DateTime, Enum, Index, Integer, JSON, String, Text, func
from sqlalchemy.dialects.mysql import INTEGER
from sqlalchemy.orm import Mapped, mapped_column
from app.models import Base, ID

UNSIGNED_COUNT = Integer().with_variant(INTEGER(unsigned=True), 'mysql')


class ReviewQueue(Base):
    __tablename__ = 'review_queue'
    __table_args__ = (Index('idx_review_status', 'review_status'),)
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    normalized_question: Mapped[str] = mapped_column(String(512))
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(
        UNSIGNED_COUNT, default=1, server_default='1'
    )
    review_status: Mapped[str] = mapped_column(
        Enum('待审', '通过', '驳回'), default='待审', server_default='待审'
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class EvalRun(Base):
    __tablename__ = 'eval_runs'
    __table_args__ = (Index('idx_created_at', 'created_at'),)
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    triggered_by: Mapped[str] = mapped_column(
        Enum('定时', '手动'), default='定时', server_default='定时'
    )
    dataset_size: Mapped[int] = mapped_column(UNSIGNED_COUNT)
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
