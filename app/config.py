"""从项目根目录的 .env 和系统环境变量读取配置。"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
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
        populate_by_name=True,
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
    # 以下旧额度供早期章节/历史图使用；正式Ch07图改走下面model_context_window推导。
    tool_context_budget: int = Field(default=8000, gt=0)
    # 这是单次工具执行超时；只读是否重试由 ToolRegistry 决定。
    tool_timeout_seconds: float = Field(default=5, gt=0, le=60)
    # 图状态使用独立 Postgres；DSN 作为 SecretStr，不能在日志中打印。
    graph_postgres_dsn: SecretStr | None = None
    # max_steps按LLM决策次数计，不是图节点数，也不是工具调用次数。
    agent_max_steps: int = Field(
        default=6,
        ge=1,
        le=20,
        validation_alias=AliasChoices('MAX_AGENT_STEPS', 'AGENT_MAX_STEPS'),
    )
    # 每次模型输出的硬上限，区别于前面控制输入上下文的tool_context_budget。
    agent_output_token_limit: int = Field(
        default=600,
        ge=1,
        le=4000,
        validation_alias=AliasChoices('MAX_OUTPUT_TOKENS', 'AGENT_OUTPUT_TOKEN_LIMIT'),
    )
    # Ch05旧图的累计调用额度；Ch07仍记累计用量，但执行限制取窗口、峰值、步数和超时。
    agent_total_token_budget: int = Field(default=24000, gt=0)
    # 覆盖分类、检索及全部Agent/工具步的整轮墙钟超时，不只限制一个工具。
    workflow_timeout_seconds: float = Field(default=180, gt=0, le=600)
    # 默认沿用已确认聊天模型；小模型名只有显式配置后才启用一次升级路线。
    intent_confidence_threshold: float = Field(
        default=0.65, ge=0, le=1, allow_inf_nan=False
    )
    intent_small_model: str | None = None
    # Ch07运行窗口必须与上游能力核对。默认对应当前flash官方1M窗口；可保守下调。
    model_context_window: int = Field(default=1_000_000, gt=0)
    # 单轮硬上限同时参与倒推：用户原话由API拦截；过大工具结果保留原State但停止后续模型。
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    # Ch08动态工具定义需要独立预留；旧Ch07演示公式仍使用原context_system_reserve。
    # 预留值不是实测大小；启动及每执行段都再核验实际System+Schema，超出就明确失败。
    tool_system_reserve: int = Field(default=3000, gt=0)
    rerank_top_k: int = Field(default=10, ge=1, le=50)
    # 预留开销需和实际输入同步：prefix启动自检，证据/摘要在装配时按整块控制。
    context_system_reserve: int = Field(default=1200, gt=0)
    context_evidence_per_hit: int = Field(default=350, gt=0)
    context_summary_budget: int = Field(default=1000, gt=0)
    context_safety_margin: int = Field(default=800, gt=0)
    context_intermediate_tokens: int = Field(default=600, gt=0)
    # “想留的量”和“装得下的量”取小；R×S不是模型窗口，也不是消息行数阈值。
    context_target_rounds: int = Field(default=64, ge=1)
    context_steady_round_tokens: int = Field(default=2000, gt=0)
    context_assistant_prefix_chars: int = Field(default=60, ge=1)
    # 与保留预算共同校准；名字虽叫ASCII，实际计数器将所有非CJK字符归入此口径。
    context_ascii_chars_per_token: float = Field(default=4.0, gt=0, allow_inf_nan=False)


@lru_cache
# 无参数函数缓存首次加载的配置；修改 .env 后需要重启服务。
def get_settings() -> Settings:
    """首次调用时加载配置，之后复用同一个实例。"""
    return Settings()
