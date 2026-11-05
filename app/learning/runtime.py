"""应用共享飞轮与定时任务；待归并事实留SQL，周期任务不在重启时立即付费重跑。"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.learning.flywheel import Flywheel
from app.learning.observability import Tracing, ObservabilitySettings
from app.learning.evaluation import EvaluationRunner
from langchain_core.runnables import RunnableLambda

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def next_evaluation(started, latest, hours):
    interval = hours * 3600
    return (
        started + interval if latest is None else max(started + 60, latest + interval)
    )


class LearningSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='LEARNING_', env_file=ROOT / '.env.observability', extra='ignore'
    )
    flywheel_enabled: bool = True
    evaluation_enabled: bool = False
    evaluation_hours: float = Field(default=24, gt=0)
    worker_seconds: float = Field(default=10, ge=1)


class LearningRuntime:
    def __init__(self, db, knowledge, model, *, testing=False, model_name='configured'):
        self.settings = LearningSettings()
        self.tracing = Tracing(
            ObservabilitySettings(enabled=False, _env_file=None) if testing else None
        )
        self.profile = json.loads(
            (ROOT / 'config/evidence-confidence.json').read_text(encoding='utf-8')
        )
        self.assessor = None
        self.flywheel = Flywheel(db, knowledge, model)
        self.evaluation = EvaluationRunner(
            db, knowledge, model, self.profile, model_name=model_name
        )
        # 后台不是主图调用：各自一次绑定外层Runnable，内部模型自动继承，避免逐模型打补丁。
        callbacks = [self.tracing.handler] if self.tracing.handler else []
        self.worker_chain = RunnableLambda(self._worker_step).with_config(
            callbacks=callbacks
        )
        self.eval_chain = RunnableLambda(self._eval_step).with_config(
            callbacks=callbacks
        )
        self.testing = testing
        self.stop = asyncio.Event()
        self.tasks = set()
        self.evaluation_task = None
        self.last_error = None

    async def start(self):
        if self.testing:
            return
        self._track(asyncio.create_task(self._loop()))

    def _track(self, task):
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def start_evaluation(self):
        if self.evaluation_task and not self.evaluation_task.done():
            raise ValueError('已有评估轮次运行中')
        self.evaluation_task = self._track(asyncio.create_task(self._evaluate('手动')))
        return dict(started=True)

    async def _evaluate(self, trigger):
        try:
            with self.tracing.request(
                'evaluation-run', {'trigger': trigger}, 'evaluation'
            ):
                self.tracing.intent('后台评估', 'general')
                return await self.eval_chain.ainvoke(trigger)
        except Exception as exc:
            self.last_error = '评估：' + type(exc).__name__
            logger.warning('Ch09评估失败 type=%s', type(exc).__name__)

    async def _worker_step(self, _):
        return await self.flywheel.process()

    async def _eval_step(self, trigger):
        return await self.evaluation.run(trigger)

    async def _loop(self):
        # 新启动没有历史轮次时从启动时间等待一个间隔，不在每次重启立即扣费。
        started = datetime.now(timezone.utc).timestamp()
        next_eval = next_evaluation(started, None, self.settings.evaluation_hours)
        try:
            runs = await self.evaluation.runs()
            if runs:
                latest = (
                    datetime.fromisoformat(runs[-1]['created_at'])
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
                next_eval = next_evaluation(
                    started, latest, self.settings.evaluation_hours
                )
        except Exception as exc:
            self.last_error = '读取轮次：' + type(exc).__name__
        while not self.stop.is_set():
            if self.settings.flywheel_enabled:
                try:
                    if await self.flywheel.has_pending():
                        with self.tracing.request('flywheel-worker', {}, 'flywheel'):
                            self.tracing.intent('后台飞轮', 'general')
                            await self.worker_chain.ainvoke({})
                except Exception as exc:
                    self.last_error = '归并：' + type(exc).__name__
                    logger.warning('Ch09归并失败 type=%s', type(exc).__name__)
            now = datetime.now(timezone.utc).timestamp()
            if self.settings.evaluation_enabled and now >= next_eval:
                next_eval = now + self.settings.evaluation_hours * 3600
                if not self.evaluation_task or self.evaluation_task.done():
                    self.evaluation_task = self._track(
                        asyncio.create_task(self._evaluate('定时'))
                    )
            try:
                await asyncio.wait_for(
                    self.stop.wait(), timeout=self.settings.worker_seconds
                )
            except TimeoutError:
                pass

    async def close(self):
        self.stop.set()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.tracing.close()
