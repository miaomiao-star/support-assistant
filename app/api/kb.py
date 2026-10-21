"""知识库管理接口；仅供本机开发，业务服务由应用生命周期注入。"""
from dataclasses import asdict
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from app.knowledge.materials import Materials, manual_chunks
from app.knowledge.runtime import ROOT, build_miner

router=APIRouter(prefix='/kb',tags=['knowledge'])
materials=Materials(ROOT/'data'/'kb')


# 导入请求只带来源名称和预览摘要；服务端重新读取文件，不信任客户端提供正文。
class ImportRequest(BaseModel):
    name: str
    digest: str


# 表单边界校验在接口层完成；具体切分和防重复交给材料/仓库层。
class ManualRequest(BaseModel):
    model_config=ConfigDict(str_strip_whitespace=True,extra='forbid')
    category: str=Field(min_length=1,max_length=255)
    questions: str=Field(min_length=1,max_length=1000)
    answer: str=Field(min_length=1,max_length=50000)
    section_path: str=Field(default='',max_length=400)
    content_type: Literal['faq','policy','manual']='faq'
    is_key_clause: bool=False


class SearchRequest(BaseModel):
    query: str=Field(min_length=1,max_length=64)


def fail(exc):
    # 只回显本地校验错误；上游异常不得泄漏连接地址或请求头。
    raise HTTPException(status_code=409,detail=str(exc)) from None


# 列表统计按来源路径对应知识块数，不触发向量化。
@router.get('/materials')
async def list_materials(request:Request):
    rows=await request.app.state.knowledge.store.all_chunks()
    return [{'name':name,'imported':sum((r['section_path'] or '').startswith(name+' / ') for r in rows)} for name in materials.names()]


# 只计算切分结果和摘要，用户确认后才调用import写库。
@router.get('/preview')
async def preview(name:str):
    try: return materials.preview(name)
    except ValueError as exc: fail(exc)


# 先验证预览摘要，再原子写入同来源所有块；重复提交由仓库复用原主键。
@router.post('/import')
async def import_material(body:ImportRequest,request:Request):
    try: return await request.app.state.knowledge.store.import_chunks(materials.checked(body.name,body.digest),body.name)
    except ValueError as exc: fail(exc)


# 与保存共用切分函数，保证页面看到的预览和实际写入一致。
@router.post('/manual/preview')
async def preview_manual(body:ManualRequest):
    try:
        chunks,source=manual_chunks(body.model_dump())
        return {'chunks':[asdict(c) for c in chunks]}
    except ValueError as exc: fail(exc)


# 仅保存原文为pending；本次请求不等待嵌入模型完成。
@router.post('/manual')
async def save_manual(body:ManualRequest,request:Request):
    try:
        chunks,source=manual_chunks(body.model_dump())
        return await request.app.state.knowledge.store.import_chunks(chunks,source)
    except ValueError as exc: fail(exc)


# counts来自数据库事实，job来自后台任务快照，两者用途不同。
@router.get('/status')
async def status(request:Request):
    return {'counts':await request.app.state.knowledge.store.stats(),'job':request.app.state.kb_jobs.status()}


# 页面mine只抽取去重；向量化是另一项任务，调度入口则会串联两者。
@router.post('/jobs/{kind}')
async def start_job(kind:Literal['vectorize','mine'],request:Request):
    service=request.app.state.knowledge
    try:
        worker=service.vectorize if kind=='vectorize' else build_miner(service).run
        return request.app.state.kb_jobs.start(kind,worker)
    except ValueError as exc: fail(exc)


@router.post('/stop')
async def stop_job(request:Request): return request.app.state.kb_jobs.stop()


# diagnostics额外返回分数与来源；聊天工具不暴露这些管理页字段。
@router.post('/search')
async def search(body:SearchRequest,request:Request):
    try: return {'data':await request.app.state.knowledge.search(body.query,diagnostics=True)}
    except Exception:
        raise HTTPException(status_code=502,detail='检索服务调用失败，请检查嵌入配置与 Lite 服务') from None


@router.get('/staging')
async def staging(request:Request):
    rows=await request.app.state.knowledge.store.staging()
    # 读取运行清单不构造模型，不触发付费调用。
    import json
    path=ROOT/'work'/'ch03'/'mining.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    # 把本地审核原因关联到暂存行；数据库状态仍取自staging表。
    for row in rows: row['review']=state.get('decisions',{}).get(str(row['id']))
    return rows
