from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.config import get_settings
from app.core.memory import SessionStore
from app.db import Database, DatabaseSettings


def create_app(database: Database | None = None) -> FastAPI:
    """组装应用；工厂形式让测试能创建彼此独立的应用和会话存储。"""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """
        启动时加载并校验配置；这里只校验字段，不请求模型验证密钥或模型名。
        """
        get_settings()
        # 测试可传入独立 database；真实启动读取 .env.mysql 创建异步 MySQL 引擎。
        # 挂到 app.state 的是应用共享资源，不是某个浏览器的聊天历史。
        application.state.database = database or Database(DatabaseSettings().url())
        try:
            # 只检查连通性，不在应用启动时重建业务表。
            await application.state.database.check()
            yield
        finally:
            # 正常关闭或启动检查失败，都清理已经创建的数据库引擎。
            await application.state.database.close()
    # lifespan是FastAPI的生命周期管理器，允许在应用启动和关闭时执行特定的代码。
    application = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
    application.state.sessions = SessionStore()
    # SessionStore 保留第一章接口便于回归；新聊天流程只使用它的 lock()。
    # 数据库历史恢复发生在 ChatRepository.history()，不再调用这里的 commit()。
    # app.state 是应用级共享对象，不是每个请求各建一份。
    # 这里只使用单进程会话锁；真实聊天历史由数据库仓库读取。
    application.include_router(chat_router, prefix="/api", tags=["chat"])
    application.include_router(extract_router, prefix="/api", tags=["extract"])
    application.frontend("/", directory=Path(__file__).resolve().parent / "static")
    # 静态页与 API 同源，前端可用 /api/chat，无需跨域配置。
    return application


# uvicorn app.main:app 中冒号左边是模块，右边是这个应用对象。
app = create_app()
