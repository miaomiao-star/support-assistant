import asyncio
import pytest
from app.db import Database
from app.models import Base
from app.knowledge.store import KnowledgeStore
from app.knowledge.splitter import Chunk
from app.knowledge.service import KnowledgeService


# 固定三维向量仅用于流程测试，不验证真实语义质量，也不产生模型费用。
class Embeddings:
    async def aembed_documents(self, texts): return [[1.,0.,0.] for _ in texts]
    async def aembed_query(self, text): return [1.,0.,0.]


# 内存字典模拟同主键upsert；fail在写入后抛错，复现最危险的双写窗口。
class Vectors:
    def __init__(self): self.rows={}; self.fail=False
    async def upsert(self, key, vector):
        self.rows[key]=vector
        if self.fail: raise ConnectionError('写入后连接断开')
    async def search(self, vector, limit): return [(k,.9) for k in self.rows][:limit]
    async def close(self): pass


def test_import_idempotence_links_and_crash_repair(tmp_path):
    """假向量库模拟“写成功但响应丢失”，验证pending不提前done及重试不增行。"""
    async def run():
        db=Database('sqlite+aiosqlite://')
        async with db.engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
        store=KnowledgeStore(db)
        rows=[Chunk('政策','退货',a,'x.md / 政策','policy') for a in ['第一句。','第二句。']]
        first=await store.import_chunks(rows,'x.md')
        second=await store.import_chunks(rows,'x.md')
        assert first['created']==2 and second['created']==0
        records=await store.all_chunks()
        assert records[0]['next_chunk_id']==records[1]['id']
        assert records[1]['prev_chunk_id']==records[0]['id']
        with pytest.raises(ValueError): await store.import_chunks(rows[:1],'x.md')
        vec=Vectors(); svc=KnowledgeService(store,Embeddings(),vec)
        vec.fail=True
        result=await svc.vectorize()
        assert result['failed']==2 and (await store.stats())['pending']==2
        assert await svc.search('邮费')==[]
        vec.fail=False
        assert (await svc.vectorize())['succeeded']==2
        assert len(vec.rows)==2 and (await store.stats())['pending']==0
        assert len(await svc.search('邮费'))==2
        await db.close()
    asyncio.run(run())


def test_stop_leaves_pending_for_resume():
    """预先设置停止信号，验证零写入且可以再次补齐。"""
    async def run():
        db=Database('sqlite+aiosqlite://')
        async with db.engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
        store=KnowledgeStore(db)
        await store.import_chunks([Chunk('类','问','答。','a / b','faq')],'a')
        stop=asyncio.Event(); stop.set()
        svc=KnowledgeService(store,Embeddings(),Vectors())
        assert (await svc.vectorize(stop))['succeeded']==0
        assert (await store.stats())['pending']==1
        assert (await svc.vectorize())['succeeded']==1
        await db.close()
    asyncio.run(run())
