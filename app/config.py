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
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 中未声明的字段不导致加载失败。
    )

    chat_api_key: SecretStr = Field(validation_alias="CHAT_API_KEY", min_length=1)
    # validation_alias 映射 .env 名称；代码使用小写属性名。
    # SecretStr 打印时隐藏密钥，但不是加密，也不能代替权限管理。
    chat_base_url: str = Field(validation_alias="CHAT_BASE_URL", min_length=1)
    chat_model: str = Field(validation_alias="CHAT_MODEL", min_length=1)
    # 本章未启用 embedding / rerank，因此这两组允许不填写。
    embed_api_key: SecretStr | None = Field(default=None, validation_alias="EMBED_API_KEY")
    embed_base_url: str | None = Field(default=None, validation_alias="EMBED_BASE_URL")
    embed_model: str | None = Field(default=None, validation_alias="EMBED_MODEL")
    rerank_api_key: SecretStr | None = Field(default=None, validation_alias="RERANK_API_KEY")
    rerank_base_url: str | None = Field(default=None, validation_alias="RERANK_BASE_URL")
    rerank_model: str | None = Field(default=None, validation_alias="RERANK_MODEL")
    llm_token_budget: int = Field(default=2000, validation_alias="LLM_TOKEN_BUDGET", gt=0)
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


@lru_cache
# 无参数函数缓存首次加载的配置；修改 .env 后需要重启服务。
def get_settings() -> Settings:
    """首次调用时加载配置，之后复用同一个实例。"""
    return Settings()
