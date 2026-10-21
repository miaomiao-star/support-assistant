import asyncio
from unittest.mock import AsyncMock
import pytest
from app.db import Database
from app.models import Base
from app.knowledge.store import KnowledgeStore
from app.knowledge.models import KnowledgeChunk
from app.knowledge.splitter import Chunk
from app.knowledge.mining import structured, Decision
from langchain_core.exceptions import OutputParserException


def test_reimport_after_adding_mined_question_keeps_source_identity():
    """追加真实问法后重新导入原材料，仍应识别为同一来源。"""
    async def run():
        db=Database('sqlite+aiosqlite://')
        async with db.engine.begin() as c: await c.run_sync(Base.metadata.create_all)
        store=KnowledgeStore(db)
        chunks=[Chunk('物流','运费怎么算','满99包邮。','a.md / 运费','faq')]
        result=await store.import_chunks(chunks,'a.md')
        async with db.sessions.begin() as s:
            row=await s.get(KnowledgeChunk,result['ids'][0]); row.questions+='\n邮费呢'
        assert (await store.import_chunks(chunks,'a.md'))['created']==0
        await db.close()
    asyncio.run(run())


def test_json_format_repair_is_bounded():
    """格式错误允许纠正一次；连续失败不得无限请求模型。"""
    async def run():
        chain=AsyncMock(); chain.ainvoke.side_effect=[OutputParserException('bad'),Decision(decision='new',category='演示',reason='新知识')]
        class Model:
            def with_structured_output(self,*args,**kwargs): return chain
        assert (await structured(Model(),Decision,'输出JSON',{})).decision=='new'
        assert chain.ainvoke.await_count==2
        chain.ainvoke.reset_mock();chain.ainvoke.side_effect=OutputParserException('bad')
        with pytest.raises(OutputParserException): await structured(Model(),Decision,'输出JSON',{})
        assert chain.ainvoke.await_count==2
    asyncio.run(run())


def test_vector_cancellation_waits_for_write_thread():
    """取消协程后必须等写线程结束，才能向外报告取消并释放写锁。"""
    import threading
    from app.knowledge.vector import MilvusVectors
    async def run():
        vectors=MilvusVectors('unused')
        entered=threading.Event(); release=threading.Event(); completed=threading.Event()
        def operation():
            entered.set(); release.wait(timeout=3); completed.set()
        task=asyncio.create_task(vectors._call(operation))
        while not entered.is_set(): await asyncio.sleep(.005)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError): await task
        assert completed.is_set()
    asyncio.run(run())
