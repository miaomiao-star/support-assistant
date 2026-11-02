from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.config import get_settings
from app.core.memory import SessionStore
from app.db import Database, DatabaseSettings
from app.api.kb import router as kb_router
from app.knowledge.runtime import build_service, ROOT
from app.knowledge.jobs import Jobs
from fastapi.responses import FileResponse
from app.workflow.conversation import build_graph, prefix_tokens, Context
from app.context.budget import ContextBudget
from app.context.summary import SummaryTasks
from app.context.observability import setup_logging, event
from app.tools.registry import ToolRegistry
from app.workflow.persistence import open_checkpointer


def create_app(
    database: Database | None = None,
    *,
    checkpointer=None,
    workflow_enabled=True,
    graph_factory=None,
) -> FastAPI:
    """组装Web应用、共享图和资源生命周期；测试可显式注入隔离依赖。

    正常入口保持workflow_enabled=True并打开官方Postgres saver。
    checkpointer注入和关闭Workflow供测试使用，不能通过聊天请求切换。
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """
        启动时加载并校验配置；这里只校验字段，不请求模型验证密钥或模型名。
        """
        settings = get_settings()
        # Ch07原文上下文写独立本地日志；只配置handler，不在启动时打印私有Settings。
        setup_logging(Path(__file__).resolve().parent.parent)
        # 测试可传入独立 database；真实启动读取 .env.mysql 创建异步 MySQL 引擎。
        # 挂到 app.state 的是应用共享资源，不是某个浏览器的聊天历史。
        application.state.database = database or Database(DatabaseSettings().url())
        # 本章知识服务在应用内共享，聊天与管理页使用相同模型、向量库和阈值。
        # 显式传入测试数据库时保留隔离的Ch03假依赖；真实应用按迁移开关选择索引。
        application.state.knowledge = build_service(
            application.state.database, rag_enabled=False if database else None
        )
        application.state.rag = (
            application.state.knowledge
            if application.state.knowledge.retriever
            else None
        )
        application.state.kb_jobs = Jobs(
            ROOT / "work" / "ch03" / ("test" if database else "web")
        )
        try:
            # 只检查连通性，不在应用启动时重建业务表。
            await application.state.database.check()
            # ExitStack托管动态创建的异步资源：初始化图失败或Web退出都会关闭连接池。
            async with AsyncExitStack() as resources:
                if workflow_enabled:
                    saver = checkpointer
                    if saver is None:
                        saver = await resources.enter_async_context(
                            open_checkpointer(get_settings())
                        )
                    # 编译图只做节点/边/状态与saver的组装；这里不发起LLM对话。
                    # 图由整个应用共享，真正的请求依赖通过Runtime Context逐轮传入。
                    application.state.workflow = (graph_factory or build_graph)(saver)
                    if getattr(application.state.workflow, 'layered_context', False):
                        # 历史图不套本章预算；正式图先验证公式，再核验真实System+工具Schema额度。
                        budget = ContextBudget.from_settings(settings)
                        measured = prefix_tokens(
                            Context(
                                None, None, None, settings, ToolRegistry(None, 0, ''), 0
                            )
                        )
                        # 临时注册表只为读取Schema，None仓库不会被执行；这一步不请求任何模型。
                        if measured > settings.context_system_reserve:
                            raise ValueError(
                                '上下文预算不足：系统和工具定义超过配置保留量，请一起校准预算'
                            )
                        event(
                            'context startup',
                            history=budget.history,
                            layer1=budget.layer1,
                            layer2=budget.layer2,
                            react_peak=budget.react_peak,
                            prefix_tokens=measured,
                            system_reserve=settings.context_system_reserve,
                            zh_chars_pre_token=settings.zh_chars_pre_token,
                            ascii_chars_per_token=settings.context_ascii_chars_per_token,
                        )
                        application.state.summary_tasks = SummaryTasks()
                        # 应用只建一份任务注册表，HTTP每轮只注入引用，才能按会话去重后台任务。
                yield
        finally:
            # 先收敛后台任务，再关闭其依赖的向量连接和数据库，避免使用已关闭资源。
            if getattr(application.state, 'summary_tasks', None):
                await application.state.summary_tasks.close()
            await application.state.kb_jobs.close()
            await application.state.knowledge.close()
            # 正常关闭或启动检查失败，都清理已经创建的数据库引擎。
            await application.state.database.close()

    # lifespan是FastAPI的生命周期管理器，允许在应用启动和关闭时执行特定的代码。
    application = FastAPI(title="MewHelp", version="0.1.0", lifespan=lifespan)
    application.state.sessions = SessionStore()
    # 生产默认只走 LangGraph；显式关闭仅供旧章节适配器回归使用。
    application.state.workflow = None
    # SessionStore 保留第一章接口便于回归；新聊天流程只使用它的 lock()。
    # Ch07从checkpoint延续完整messages；只把SQL中尚未见过的文本审计补入，不替换工具流水。
    # app.state 是应用级共享对象，不是每个请求各建一份。
    # 这里只使用单进程会话锁；checkpointer持久化不等于已实现多副本同会话互斥。
    application.include_router(chat_router, prefix="/api", tags=["chat"])
    application.include_router(extract_router, prefix="/api", tags=["extract"])
    application.include_router(kb_router, prefix="/api")

    @application.get("/kb", include_in_schema=False)
    async def knowledge_page():
        """明确页面路由先于首页静态回退，避免 /kb 被首页吞掉。"""
        return FileResponse(Path(__file__).resolve().parent / "static" / "kb.html")

    @application.get("/kb/source/{key}", include_in_schema=False)
    async def source_page(key: int):
        # 页面模板不接收文件路径，浏览器再按数字主键向原文API读取内容。
        return FileResponse(Path(__file__).resolve().parent / "static" / "source.html")

    application.frontend("/", directory=Path(__file__).resolve().parent / "static")
    # 静态页与 API 同源，前端可用 /api/chat，无需跨域配置。
    return application


# uvicorn app.main:app 中冒号左边是模块，右边是这个应用对象。
app = create_app()
