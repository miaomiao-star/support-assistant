from datetime import datetime, timezone
from types import SimpleNamespace as R
from app.knowledge.mining import complete_turns, safe_qa, next_run


def test_only_complete_turns_without_mock():
    """仅完整、非模拟回合可成为抽取来源。"""
    rows=[R(id=1,conversation_id=1,role='user',content='退货要求',tool_calls=None),
          R(id=2,conversation_id=1,role='assistant',content='完整商品七天内退货',tool_calls=None),
          R(id=3,conversation_id=2,role='user',content='订单1物流',tool_calls=None),
          R(id=4,conversation_id=2,role='tool',content='{"mock":true}',tool_calls=None),
          R(id=5,conversation_id=2,role='assistant',content='模拟配送',tool_calls=None),
          R(id=6,conversation_id=1,role='user',content='没有回复',tool_calls=None)]
    result=complete_turns(rows)
    assert len(result)==1 and result[0]['source_ref']=='conversation:1:answer:2'


def test_extraction_must_be_grounded_and_no_personal_information():
    """模型改写或无依据扩写不能通过出处校验。"""
    source={'question':'可以退货吗','answer':'支持七天退货。','source_ref':'a'}
    assert safe_qa({'question':'可以退货吗','answer':'支持七天退货。','source_ref':'a'},[source])
    assert not safe_qa({'question':'能退吗','answer':'随时无条件退货','source_ref':'a'},[source])
    assert not safe_qa({'question':'可以退货吗','answer':'请拨打13800138000','source_ref':'a'},[source])


def test_next_schedule_is_shanghai_two_am():
    """用UTC构造北京时间02:00前后边界，不依赖测试机当前时间。"""
    now=datetime(2026,10,20,17,59,tzinfo=timezone.utc)
    assert (next_run(now)-now).total_seconds()==60
    now=datetime(2026,10,20,18,0,tzinfo=timezone.utc)
    assert (next_run(now)-now).total_seconds()==86400


def test_staging_transaction_dedupe_conflict_and_replay(tmp_path):
    """固定模型判断验证事务/状态与重放，不把假模型结果当成Prompt质量评估。"""
    import asyncio,json
    from app.db import Database
    from app.models import Base,Message,Conversation
    from app.knowledge.models import QAStaging
    from app.knowledge.mining import Miner,Extracted,ExtractedQA,Decision
    from app.knowledge.store import KnowledgeStore
    from app.knowledge.splitter import Chunk
    from test_kb_store import Embeddings
    class Model:
        def with_structured_output(self,schema,**kwargs):
            class Chain:
                async def ainvoke(self,messages):
                    data=json.loads(messages[-1][1])
                    if schema is Extracted: return Extracted(items=[ExtractedQA(**r) for r in data])
                    q=data['incoming']['question']
                    return Decision(decision='conflict' if q=='运费呢' else 'new',
                        match_id=data['candidates'][0]['id'] if q=='运费呢' else None,category='演示',reason='标注判断')
            return Chain()
    async def run():
        db=Database('sqlite+aiosqlite://')
        async with db.engine.begin() as c: await c.run_sync(Base.metadata.create_all)
        store=KnowledgeStore(db)
        await store.import_chunks([Chunk('物流','邮费','收10元。','a / b','faq')],'a')
        async with db.sessions.begin() as s:
            s.add(Conversation(id=1,user_id='test')); await s.flush()
            for q,a in [('邮费','收10元。'),('礼品袋','下单备注即可。'),('运费呢','收6元。')]:
                s.add(Message(conversation_id=1,role='user',content=q))
                s.add(Message(conversation_id=1,role='assistant',content=a))
        miner=Miner(store,Embeddings(),Model(),tmp_path)
        result=await miner.run()
        assert result['kept']==1 and result['discarded']==1 and result['conflicts']==1
        assert {r['status'] for r in await store.staging()}=={'kept','discarded','extracted'}
        assert len(await store.all_chunks())==2
        again=await miner.run()
        assert again['extracted']==0 and again['kept']==0
        assert len(await store.all_chunks())==2 and len(await store.staging())==3
        await db.close()
    asyncio.run(run())
