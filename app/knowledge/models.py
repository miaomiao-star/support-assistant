"""严格映射用户 ch03-ddl.sql；线上建表仍使用原 SQL。"""
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from app.models import Base, ID


class KnowledgeChunk(Base):
    # 正式知识表是原文权威源；向量库只存关联主键与向量。
    __tablename__='knowledge_chunks'
    id: Mapped[int]=mapped_column(ID,primary_key=True,autoincrement=True)
    category: Mapped[str]=mapped_column(String(255),index=True)
    questions: Mapped[str]=mapped_column(Text)
    answer: Mapped[str]=mapped_column(Text)
    # 以下四类元数据不进入 embedding，也暂不触发前后块扩展召回。
    section_path: Mapped[str | None]=mapped_column(String(512))
    content_type: Mapped[str | None]=mapped_column(String(32))
    is_key_clause: Mapped[bool]=mapped_column(Boolean,default=False)
    prev_chunk_id: Mapped[int | None]=mapped_column(ID,ForeignKey('knowledge_chunks.id',ondelete='SET NULL'))
    next_chunk_id: Mapped[int | None]=mapped_column(ID,ForeignKey('knowledge_chunks.id',ondelete='SET NULL'))
    # vector_id 使用本行 id 的字符串形式；仅在向量写入成功后回填。
    vector_id: Mapped[str | None]=mapped_column(String(64))
    vectorize_status: Mapped[str]=mapped_column(Enum('pending','done'),default='pending',index=True)
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
    updated_at: Mapped[datetime]=mapped_column(DateTime,server_default=func.now(),onupdate=func.now())


class QAStaging(Base):
    # 抽取结果先在此留痕，审核通过后再新增或合并正式知识。
    __tablename__='qa_extraction_staging'
    id: Mapped[int]=mapped_column(ID,primary_key=True,autoincrement=True)
    batch_no: Mapped[str]=mapped_column(String(64),index=True)
    source_ref: Mapped[str | None]=mapped_column(String(255))
    question: Mapped[str]=mapped_column(Text)
    answer: Mapped[str]=mapped_column(Text)
    # 冲突仍保留 extracted；kept 表示新知识保留，discarded 表示重复处理。
    # 审核原因另存 work/ch03/mining.json，不擅自增加用户 DDL 的状态。
    status: Mapped[str]=mapped_column(Enum('extracted','kept','discarded'),default='extracted',index=True)
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
