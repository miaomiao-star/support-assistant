from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.config import get_settings
from app.core.memory import SessionStore


def create_app() -> FastAPI:
    """组装应用；工厂形式让测试能创建彼此独立的应用和会话存储。"""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # 启动时加载并校验配置；这里只校验字段，不请求模型验证密钥或模型名。
        get_settings()
        # yield 前是启动阶段，后是关闭阶段。本章没有额外关闭逻辑。
        yield
    # lifespan是FastAPI的生命周期管理器，允许在应用启动和关闭时执行特定的代码。
    application = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
    application.state.sessions = SessionStore()
    # app.state 是应用级共享对象，不是每个请求各建一份。
    # 每个进程各有自己的 dict；重启丢失，多 worker 无法共享历史。
    application.include_router(chat_router, prefix="/api", tags=["chat"])
    application.include_router(extract_router, prefix="/api", tags=["extract"])
    application.frontend("/", directory=Path(__file__).resolve().parent / "static")
    # 静态页与 API 同源，前端可用 /api/chat，无需跨域配置。
    return application


# uvicorn app.main:app 中冒号左边是模块，右边是这个应用对象。
app = create_app()
