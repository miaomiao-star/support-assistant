"""按用户ch04-ddl.sql映射；真实建表执行该SQL，不自动改写既有表。"""

from datetime import datetime
from sqlalchemy import String, Text, DateTime, Enum, JSON, ForeignKey, Integer, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from app.models import Base, ID
from app.learning.models import ReviewQueue


# 此表记录真实聊天的拒答轮次；评估脚本不把测试问题写进这里。
class LowConfidenceQuestion(Base):
    __tablename__ = 'low_confidence_questions'
    __table_args__ = (Index('idx_matched_review_id', 'matched_review_id'),)
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ID, ForeignKey('conversations.id')
    )
    # Ch09启用第三入口；来源与当轮快照一起保留，模型标准化失败也不丢原话。
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum('retrieval_low_conf', 'self_check', 'user_feedback'), index=True
    )
    reason: Mapped[str | None] = mapped_column(Text)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    matched_review_id: Mapped[int | None] = mapped_column(
        ID, ForeignKey('review_queue.id', name='fk_lcq_review', ondelete='SET NULL'), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )


# 一道评估题一行，保存最近一次失败答案；历次出现次数与处置时间跨轮保留。
class FaithCase(Base):
    __tablename__ = 'faith_cases'
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default='hybrid_rerank')
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    # 保存当时送给模型的完整证据快照，避免日后原文改变后无法复盘。
    citations: Mapped[list | None] = mapped_column(JSON)
    judge_model: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum('未解决', '已解决', '无需解决'), default='未解决', index=True
    )
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), index=True
    )
    resolution: Mapped[str | None] = mapped_column(String(300))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
