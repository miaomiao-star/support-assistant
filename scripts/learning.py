"""Ch09显式迁移、归并、评估与成本命令；模型/数据库配置仍沿用现有私有文件。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.db import Database, DatabaseSettings
from app.learning.migration import migrate
from app.learning.runtime import LearningRuntime
from app.learning.observability import aggregate_costs
from app.knowledge.runtime import build_service
from app.core.llm import get_chat_model
from app.config import get_settings
from app.workflow.persistence import run_async


async def main(args):
    db = Database(DatabaseSettings().url())
    knowledge = runtime = None
    try:
        if args.command == 'init':
            await migrate(db, ROOT)
            print('原Ch09 DDL已执行/重复核验：两表、两列、归并外键。')
            return
        knowledge = build_service(db, rag_enabled=True)
        runtime = LearningRuntime(
            db, knowledge, get_chat_model(), model_name=get_settings().chat_model
        )
        if args.command == 'process':
            with runtime.tracing.request('flywheel-manual', {}, 'flywheel'):
                runtime.tracing.intent('后台飞轮', 'general')
                result = await runtime.worker_chain.ainvoke({})
        elif args.command == 'evaluate':
            with runtime.tracing.request('evaluation-manual', {}, 'evaluation'):
                runtime.tracing.intent('后台评估', 'general')
                result = await runtime.eval_chain.ainvoke('手动')
        elif args.command == 'costs':
            result = aggregate_costs(await runtime.tracing.observations())
        elif args.command == 'schedule':
            runtime.settings.evaluation_enabled = True
            runtime.settings.evaluation_hours = args.hours
            await runtime.start()
            print(
                '定时服务已启动；按上次轮次/间隔执行，Ctrl+C结束，不删除任务事实。',
                flush=True,
            )
            await asyncio.Event().wait()
            return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        if runtime:
            await runtime.close()
        if knowledge:
            await knowledge.close()
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'command', choices=['init', 'process', 'evaluate', 'costs', 'schedule']
    )
    parser.add_argument('--hours', type=float, default=24)
    arguments = parser.parse_args()
    if arguments.hours <= 0:
        parser.error('--hours必须大于0')
    run_async(main(arguments))
