"""依赖组装；沿用 .env 的模型地址和密钥，客户端按需创建。"""
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.core.llm import get_chat_model
from app.knowledge.store import KnowledgeStore
from app.knowledge.service import KnowledgeService
from app.knowledge.vector import MilvusVectors
from app.knowledge.mining import Miner

ROOT=Path(__file__).resolve().parents[2]


class LazyEmbeddings:
    def __init__(self): self.client=None

    # 只有真正调用嵌入服务时才创建客户端；浏览材料或看状态无需请求模型。
    def get(self):
        if self.client is None:
            settings=get_settings()
            if not settings.embed_model or not settings.embed_base_url or not settings.embed_api_key:
                raise ValueError('请配置 EMBED 模型、地址和密钥')
            if 'bge-m3' not in settings.embed_model.lower(): raise ValueError('本章必须使用 BGE-M3')
            # 沿用已配置的兼容上游，直接提交文本而不让SDK按其他模型tokenizer切块。
            # 要求返回浮点向量，保持文档和问题使用同一BGE-M3模型。
            self.client=OpenAIEmbeddings(model=settings.embed_model,base_url=settings.embed_base_url,
                api_key=settings.embed_api_key.get_secret_value(),check_embedding_ctx_length=False,
                model_kwargs={'encoding_format':'float'},request_timeout=30,max_retries=1)
        return self.client

    async def aembed_documents(self,texts): return await self.get().aembed_documents(texts)
    async def aembed_query(self,text): return await self.get().aembed_query(text)


# Web在启动时共享这一服务；离线入口也使用同样的组装方式。
def build_service(db):
    settings=get_settings()
    return KnowledgeService(KnowledgeStore(db),LazyEmbeddings(),
        MilvusVectors(settings.milvus_uri or 'http://127.0.0.1:19530'),
        threshold=settings.kb_score_threshold)


# 抽取器复用知识仓库和嵌入模型，另使用原聊天模型完成抽取/语义审核。
def build_miner(service):
    return Miner(service.store,service.embeddings,get_chat_model(),ROOT/'work'/'ch03')
