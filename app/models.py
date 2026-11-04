"""会话/消息/业务与分段摘要ORM；真实建表及变更以sql各章DDL为准。"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.mysql import BIGINT, TINYINT, INTEGER as MYSQL_INTEGER
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Base 收集继承它的表映射。定义 Python 类本身不会执行 CREATE TABLE；
# 正式 MySQL 已由 Docker 初始化 SQL 建表，单测才调用 metadata.create_all。
class Base(DeclarativeBase):
    pass


# SQLite 仅用于隔离单测；MySQL 生产映射保持 UNSIGNED BIGINT。
# with_variant 按数据库方言切换列类型；SQLite 自增主键需要 INTEGER，
# 并不是应用运行时在 MySQL 和 SQLite 之间自动切换。
ID = (
    BigInteger()
    .with_variant(BIGINT(unsigned=True), 'mysql')
    .with_variant(Integer, 'sqlite')
)


class Conversation(Base):
    # 会话壳只记录“谁的会话、处理状态、时间”，具体聊天正文在 messages。
    __tablename__ = 'conversations'
    # Mapped[int] 是 ORM 属性类型声明；mapped_column 描述对应的数据库列。
    # 本项目仓库会显式写入由 session_id 派生的稳定主键。
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    # Ch07保存匿名访客摘要用于分组；历史路径保存session摘要，均不是正式账号ID。
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    # default 由 ORM 插入时提供；真实表 SQL 中也有 DEFAULT，两者应保持一致。
    status: Mapped[str] = mapped_column(
        Enum('进行中', '已转人工', '已结束'), default='进行中'
    )
    # Ch07原DDL：summary是段的投影，两个ID划分发送视图，消息原文不搬动。
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # summary_upto是已覆盖末行；layer1_from是层2末行，两者端点都不属于下一层。
    summary_upto_msg_id: Mapped[int | None] = mapped_column(ID, nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(ID, nullable=True)
    # server_default 由数据库产生时间；onupdate 是 ORM 更新此行时带上的值。
    # DateTime 对应既有 DATETIME，不携带时区；当前没有做时间显示的时区转换。
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ConversationSummary(Base):
    # 事实段独立存表：每次新任务INSERT一段，不把旧段当输入反复压成同一条summary。
    __tablename__ = 'conversation_summaries'
    __table_args__ = (
        # seq在同一会话内唯一；范围索引用于按会话/覆盖位置定位段，不承担跨会话记忆检索。
        UniqueConstraint('conversation_id', 'seq', name='uk_conv_seq'),
        Index('idx_conv_upto', 'conversation_id', 'upto_msg_id'),
    )
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ID)
    seq: Mapped[int] = mapped_column(Integer)
    from_msg_id: Mapped[int] = mapped_column(ID)
    # from/upto为该批原始消息的闭区间；全库消息ID可有间隔，不代表逐个ID都属于本会话。
    upto_msg_id: Mapped[int] = mapped_column(ID)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Message(Base):
    # 一条记录对应协议中的一条消息，不是整个问答轮次。
    # 历史章节也存tool结果；Ch07只存user/assistant，完整tool观察单独保留在图State。
    __tablename__ = 'messages'
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    # 外键要求所属会话存在；索引帮助按会话查询流水，不会自动加载聊天历史。
    conversation_id: Mapped[int] = mapped_column(
        ID, ForeignKey('conversations.id'), index=True
    )
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
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    # 工单使用业务工单号作为主键；一次创建必须关联已经存在的会话。
    __tablename__ = 'tickets'
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ID, ForeignKey('conversations.id'), index=True
    )
    description: Mapped[str] = mapped_column(Text)
    # 类型由工具参数 Schema 限定，数据库 ENUM 再约束可存储值。
    ticket_type: Mapped[str] = mapped_column(Enum('售后', '投诉', '咨询'))
    # 创建成功只表示“待处理”，不等于人工已经解决问题或执行退款。
    status: Mapped[str] = mapped_column(Enum('待处理', '已处理'), default='待处理')
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ToolAuditLog(Base):
    # 原ch08-ddl：不挂外键，日志失败不得影响业务工具。
    __tablename__ = 'tool_audit_logs'
    __table_args__ = (
        Index('idx_conversation_id', 'conversation_id'),
        Index('idx_tool_name', 'tool_name'),
        Index('idx_status', 'status'),
    )
    id: Mapped[int] = mapped_column(ID, primary_key=True, autoincrement=True)
    # 原表允许无会话的诊断调用；没有外键，也不因会话记录不可用而阻止记审计。
    conversation_id: Mapped[int | None] = mapped_column(ID, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128))
    tool_source: Mapped[str] = mapped_column(Enum('builtin', 'mcp'))
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # 摘要由引擎截长，完整观察保留在图State；这里不承担上下文恢复。
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum('成功', '失败', '超时', '校验拦下', '权限拒绝')
    )
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 记录额外重试次数而非尝试总数；写操作为0，真实MySQL类型按原DDL使用unsigned。
    retry_count: Mapped[int] = mapped_column(
        Integer().with_variant(TINYINT(unsigned=True), 'mysql'),
        default=0,
        server_default='0',
    )
    duration_ms: Mapped[int | None] = mapped_column(
        Integer().with_variant(MYSQL_INTEGER(unsigned=True), 'mysql'), nullable=True
    )
    # 由数据库记落库时间；duration_ms另由引擎单调时钟测量，不能用两者相减代替。
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
