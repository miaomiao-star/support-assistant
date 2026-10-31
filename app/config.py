"""从项目根目录的 .env 和系统环境变量读取配置。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 从代码文件的位置找项目根目录，避免启动终端所在目录影响 .env 路径。


class Settings(BaseSettings):
    """把字符串配置转换为有类型的字段；环境变量优先于 .env。"""

    model_config = SettingsConfigDict(
        # .env.rag仅保存迁移后的非敏感开关/地址，不覆盖原模型凭据。
        # .env.graph只增加独立Postgres的私有配置；不通过源码写入密码或DSN。
        env_file=(
            PROJECT_ROOT / ".env",
            PROJECT_ROOT / ".env.rag",
            PROJECT_ROOT / ".env.graph",
        ),
        env_file_encoding="utf-8",
        extra="ignore",  # .env 中未声明的字段不导致加载失败。
    )

    chat_api_key: SecretStr = Field(validation_alias="CHAT_API_KEY", min_length=1)
    # validation_alias 映射 .env 名称；代码使用小写属性名。
    # SecretStr 打印时隐藏密钥，但不是加密，也不能代替权限管理。
    chat_base_url: str = Field(validation_alias="CHAT_BASE_URL", min_length=1)
    chat_model: str = Field(validation_alias="CHAT_MODEL", min_length=1)
    # 知识检索按需初始化embedding与rerank；浏览材料和读取状态不要求提前请求模型。
    embed_api_key: SecretStr | None = Field(
        default=None, validation_alias="EMBED_API_KEY"
    )
    embed_base_url: str | None = Field(default=None, validation_alias="EMBED_BASE_URL")
    embed_model: str | None = Field(default=None, validation_alias="EMBED_MODEL")
    rerank_api_key: SecretStr | None = Field(
        default=None, validation_alias="RERANK_API_KEY"
    )
    rerank_base_url: str | None = Field(
        default=None, validation_alias="RERANK_BASE_URL"
    )
    rerank_model: str | None = Field(default=None, validation_alias="RERANK_MODEL")
    # Lite独立服务的连接地址；缺省地址在knowledge/runtime.py中选用本机19530端口。
    milvus_uri: str | None = Field(default=None, validation_alias="MILVUS_URI")
    # dense 分数下限须结合标注集校准；不引入关键词召回或重排。
    kb_score_threshold: float = Field(default=0.5, ge=-1, le=1)
    # 默认不切索引，只有完整迁移/核验后才通过非敏感的.env.rag启用。
    rag_enabled: bool = Field(default=False, validation_alias="RAG_ENABLED")
    rag_milvus_uri: str = Field(
        default="http://127.0.0.1:19531", validation_alias="RAG_MILVUS_URI"
    )
    rag_collection: str = Field(
        default="knowledge_ch04", validation_alias="RAG_COLLECTION"
    )
    # 开发样例校准后的门槛；重排分数不是概率，最终仍须生成自评和引用校验。
    rag_min_score: float = Field(
        default=0.03, ge=0, le=1, validation_alias="RAG_MIN_SCORE"
    )
    llm_token_budget: int = Field(
        default=2000, validation_alias="LLM_TOKEN_BUDGET", gt=0
    )
    # gt=0 要求正数；这是输入上下文预算，不是输出长度上限。
    # 保留现有 .env 中的 PRE 拼写。
    zh_chars_pre_token: float = Field(
        default=1.2, validation_alias="ZH_CHARS_PRE_TOKEN", gt=0, allow_inf_nan=False
    )
    # 原预算控制文本历史；新增总预算覆盖工具协议及返回数据。
    # service 会先从总预算预留工具定义，再按较小的文本额度保留历史。
    # 这些值限制“发送给模型的输入”，并不限制回答字数或模型总运行时间。
    tool_context_budget: int = Field(default=8000, gt=0)
    # 这是单次工具执行超时；只读是否重试由 ToolRegistry 决定。
    tool_timeout_seconds: float = Field(default=5, gt=0, le=60)
    # 图状态使用独立 Postgres；DSN 作为 SecretStr，不能在日志中打印。
    graph_postgres_dsn: SecretStr | None = None
    # max_steps按LLM决策次数计，不是图节点数，也不是工具调用次数。
    agent_max_steps: int = Field(default=6, ge=1, le=20)
    # 每次模型输出的硬上限，区别于前面控制输入上下文的tool_context_budget。
    agent_output_token_limit: int = Field(default=600, ge=1, le=4000)
    # 累加Agent多次输入/输出的估算消耗；重复发出的历史同样计入。
    agent_total_token_budget: int = Field(default=24000, gt=0)
    # 覆盖分类、检索及全部Agent/工具步的整轮墙钟超时，不只限制一个工具。
    workflow_timeout_seconds: float = Field(default=180, gt=0, le=600)
    # 默认沿用已确认聊天模型；小模型名只有显式配置后才启用一次升级路线。
    intent_confidence_threshold: float = Field(
        default=0.65, ge=0, le=1, allow_inf_nan=False
    )
    intent_small_model: str | None = None


@lru_cache
# 无参数函数缓存首次加载的配置；修改 .env 后需要重启服务。
def get_settings() -> Settings:
    """首次调用时加载配置，之后复用同一个实例。"""
    return Settings()
