"""四张表的 ORM 映射，真实建表仍以 sql/ch02-ddl.sql 为准。"""
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Base 收集继承它的表映射。定义 Python 类本身不会执行 CREATE TABLE；
# 正式 MySQL 已由 Docker 初始化 SQL 建表，单测才调用 metadata.create_all。
class Base(DeclarativeBase):
    pass


# SQLite 仅用于隔离单测；MySQL 生产映射保持 UNSIGNED BIGINT。
# with_variant 按数据库方言切换列类型；SQLite 自增主键需要 INTEGER，
# 并不是应用运行时在 MySQL 和 SQLite 之间自动切换。
ID = BigInteger().with_variant(BIGINT(unsigned=True), 'mysql').with_variant(Integer, 'sqlite')


class Conversation(Base):
    # 会话壳只记录“谁的会话、处理状态、时间”，具体聊天正文在 messages。
    __tablename__ = 'conversations'
    # Mapped[int] 是 ORM 属性类型声明；mapped_column 描述对应的数据库列。
    # 本项目仓库会显式写入由 session_id 派生的稳定主键。
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    # 当前保存完整会话摘要用于碰撞核验，还不是账号系统中的真实用户 ID。
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    # default 由 ORM 插入时提供；真实表 SQL 中也有 DEFAULT，两者应保持一致。
    status: Mapped[str] = mapped_column(Enum('进行中', '已转人工', '已结束'), default='进行中')
    # server_default 由数据库产生时间；onupdate 是 ORM 更新此行时带上的值。
    # DateTime 对应既有 DATETIME，不携带时区；当前没有做时间显示的时区转换。
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Message(Base):
    # 一条记录对应协议中的一条消息，不是整个问答轮次。
    # 带工具的一轮通常为 user → assistant(申请) → tool(结果) → assistant(回答)。
    __tablename__ = 'messages'
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    # 外键要求所属会话存在；索引帮助按会话查询流水，不会自动加载聊天历史。
    conversation_id: Mapped[int] = mapped_column(ID, ForeignKey('conversations.id'), index=True)
    role: Mapped[str] = mapped_column(Enum('user', 'assistant', 'tool'))
    # 纯工具申请可能没有正文，所以允许 NULL；TEXT 的容量也不是无限的。
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    # assistant 的工具申请列表，如 [{name, args, id, type}]；JSON 保存结构。
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # tool 消息用此字段指向申请中的 id。它不是 messages.id，
    # 也不是数据库外键；协议配对和当前回合范围由服务/仓库代码控制。
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FAQ(Base):
    # 原始 FAQ 数据源。当前只对 question 做 LIKE，不对 answer 做语义检索。
    __tablename__ = 'faq'
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class Ticket(Base):
    # 工单使用业务工单号作为主键；一次创建必须关联已经存在的会话。
    __tablename__ = 'tickets'
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ID, ForeignKey('conversations.id'), index=True)
    description: Mapped[str] = mapped_column(Text)
    # 类型由工具参数 Schema 限定，数据库 ENUM 再约束可存储值。
    ticket_type: Mapped[str] = mapped_column(Enum('售后', '投诉', '咨询'))
    # 创建成功只表示“待处理”，不等于人工已经解决问题或执行退款。
    status: Mapped[str] = mapped_column(Enum('待处理', '已处理'), default='待处理')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
