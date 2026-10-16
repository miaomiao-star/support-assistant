from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """请求契约；FastAPI 在调用业务函数前执行这些校验。"""
    # ConfigDict 允许在模型级别设置配置，而不是在每个字段上重复设置。
    # str_strip_whitespace=True 会自动去除字符串两端的空白字符。
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    # session_id 只是会话索引，不是登录凭证；生产环境应校验会话归属。
    message: str = Field(min_length=1)
