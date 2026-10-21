import asyncio
from pathlib import Path
import pytest
from app.knowledge.materials import Materials, manual_chunks
from app.knowledge.jobs import Jobs
from test_api import api


def test_preview_digest_path_guard_and_manual_validation(tmp_path):
    """验证路径越级拒绝、旧预览摘要失效和手工元数据保留。"""
    (tmp_path/'a.md').write_text('# 标题\n正文。',encoding='utf-8')
    materials=Materials(tmp_path)
    preview=materials.preview('a.md')
    assert preview['chunks'][0]['answer']=='正文。'
    with pytest.raises(ValueError): materials.preview('../secret.md')
    with pytest.raises(ValueError): materials.preview(str(tmp_path/'a.md'))
    (tmp_path/'a.md').write_text('# 标题\n变化。',encoding='utf-8')
    with pytest.raises(ValueError): materials.checked('a.md',preview['digest'])
    chunks,source=manual_chunks({'category':'售后','questions':'怎么退','answer':'七天退货。','content_type':'faq','is_key_clause':True})
    assert chunks[0].category=='售后' and chunks[0].is_key_clause
    assert source.startswith('manual:')


def test_job_stop_busy_and_restart_status(tmp_path):
    """验证任务互斥、协作停止，以及磁盘状态可重新读取。"""
    async def run():
        jobs=Jobs(tmp_path)
        async def worker(stop,progress):
            await stop.wait()
            return {'stopped':True}
        jobs.start('vectorize',worker)
        with pytest.raises(ValueError): jobs.start('other',worker)
        jobs.stop()
        await jobs.task
        assert jobs.status()['state']=='stopped'
        assert Jobs(tmp_path).status()['state']=='stopped'
    asyncio.run(run())


def test_http_preview_import_manual_and_same_retrieval(api):
    """HTTP链路使用假嵌入/向量库；检查预览只读、重复保存和聊天检索字段契约。"""
    client,app,_=api
    from test_kb_store import Embeddings, Vectors
    app.state.knowledge.embeddings=Embeddings()
    app.state.knowledge.vectors=Vectors()
    before=client.get('/api/kb/status').json()['counts']
    assert client.get('/kb').status_code==200
    assert client.get('/api/kb/preview',params={'name':'../.env'}).status_code==409
    preview=client.get('/api/kb/preview',params={'name':'product-faq.md'}).json()
    assert client.get('/api/kb/status').json()['counts']==before
    body={'name':preview['name'],'digest':preview['digest']}
    assert client.post('/api/kb/import',json={**body,'digest':'stale'}).status_code==409
    assert client.post('/api/kb/import',json=body).json()['created']==15
    assert client.post('/api/kb/import',json=body).json()['created']==0
    assert client.post('/api/kb/manual',json={'category':'','questions':'a','answer':'b'}).status_code==422
    manual={'category':'演示','questions':'礼品袋怎么申请','answer':'演示：下单备注礼品袋。'}
    assert client.post('/api/kb/manual/preview',json=manual).status_code==200
    assert client.post('/api/kb/manual',json=manual).json()['created']==1
    assert client.post('/api/kb/manual',json=manual).json()['created']==0
    assert client.post('/api/kb/search',json={'query':'邮费'}).json()['data']==[]
    client.portal.call(app.state.knowledge.vectorize)
    hits=client.post('/api/kb/search',json={'query':'邮费'}).json()['data']
    assert hits and 'score' in hits[0]
    assert set(client.portal.call(app.state.knowledge.search,'邮费')[0])=={'id','question','answer','category'}
