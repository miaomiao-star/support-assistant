"""本机开发用审核/趋势接口；不把示例答案当核准答案，也不接受原话伪造反馈。"""

from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from app.learning.observability import aggregate_costs

router = APIRouter(prefix='/api/learning', tags=['learning'])


def runtime(request):
    value = getattr(request.app.state, 'learning', None)
    if value is None:
        raise HTTPException(503, '当前图未启用Ch09')
    return value


class ApprovalBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    approved_answer: str = Field(min_length=1, max_length=50000)
    category: str = Field(default='客服FAQ', min_length=1, max_length=255)
    content_type: Literal['faq', 'policy', 'manual'] = 'faq'


@router.get('/reviews')
async def reviews(
    request: Request, status: Literal['待审', '通过', '驳回'] | None = None
):
    return await runtime(request).flywheel.list(status)


@router.get('/reviews/{key}')
async def detail(key: int, request: Request):
    try:
        return await runtime(request).flywheel.detail(key)
    except LookupError:
        raise HTTPException(404, '缺口不存在') from None


@router.post('/reviews/{key}/approve')
async def approve(key: int, body: ApprovalBody, request: Request):
    try:
        return await runtime(request).flywheel.approve(
            key,
            body.approved_answer,
            category=body.category,
            content_type=body.content_type,
        )
    except LookupError:
        raise HTTPException(404, '缺口不存在') from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post('/reviews/{key}/reject')
async def reject(key: int, request: Request):
    try:
        return await runtime(request).flywheel.reject(key)
    except LookupError:
        raise HTTPException(404, '缺口不存在') from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get('/evaluations')
async def evaluations(request: Request):
    return await runtime(request).evaluation.runs()


@router.post('/evaluations')
async def evaluate(request: Request):
    try:
        return runtime(request).start_evaluation()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None


@router.get('/status')
async def status(request: Request):
    value = runtime(request)
    return dict(
        evaluation=value.evaluation.progress,
        evaluation_running=bool(
            value.evaluation_task and not value.evaluation_task.done()
        ),
        last_error=value.last_error,
        scheduled=value.settings.evaluation_enabled,
        interval_hours=value.settings.evaluation_hours,
    )


@router.get('/costs')
async def costs(request: Request):
    value = runtime(request)
    try:
        return dict(
            groups=aggregate_costs(await value.tracing.observations()),
            window_days=7,
            note='按trace所属意图汇总generation真实usage；未知usage/计价单列，不使用字符估算',
        )
    except Exception:
        raise HTTPException(
            502, '本地Langfuse统计不可用，请检查服务和项目配置'
        ) from None
