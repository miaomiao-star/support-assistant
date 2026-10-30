from pydantic import BaseModel, ConfigDict, Field


class RetrievalFilters(BaseModel):
    """只接受已建索引的元数据；禁止传入任意Milvus表达式。"""

    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    # category是文档上级标题路径，product_category才是商品品类，两者含义不同。
    category: str | None = Field(default=None, max_length=255)
    content_type: str | None = Field(default=None, max_length=255)
    product_category: str | None = Field(default=None, max_length=255)


class ChatRequest(BaseModel):
    """请求契约；FastAPI 在调用业务函数前执行这些校验。"""

    # ConfigDict 允许在模型级别设置配置，而不是在每个字段上重复设置。
    # str_strip_whitespace=True 会自动去除字符串两端的空白字符。
    model_config = ConfigDict(str_strip_whitespace=True)

    session_id: str = Field(min_length=1, max_length=128)
    # session_id 只是会话索引，不是登录凭证；生产环境应校验会话归属。
    message: str = Field(min_length=1)
    # default_factory为每个请求创建独立对象，旧客户端省略filters时仍可工作。
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
