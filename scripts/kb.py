"""Ch03 离线入口：原 SQL 初始化、材料入库、补向量、抽取及北京时间调度。"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent.parent
# 允许从脚本路径运行时导入项目app包；配置仍由项目自身读取。
sys.path.insert(0,str(ROOT))
from sqlalchemy import inspect
from app.db import Database, DatabaseSettings
from app.knowledge.runtime import build_service, build_miner
from app.knowledge.materials import Materials
from app.knowledge.mining import next_run, SHANGHAI


async def initialize(db):
    """仅创建缺失的新表；保留原 DDL，不重新建已有表或覆盖业务数据。"""
    sql=(ROOT/'sql'/'ch03-ddl.sql').read_text(encoding='utf-8')
    import re
    sql=re.sub(r'^\s*--.*$','',sql,flags=re.M)
    async with db.engine.begin() as conn:
        # 先检查现有表，再执行用户本章DDL；不是通用SQL迁移引擎。
        tables=set(await conn.run_sync(lambda c:inspect(c).get_table_names()))
        for statement in sql.split(';'):
            statement=statement.strip()
            if not statement: continue
            match=re.search(r'CREATE TABLE (\w+)',statement)
            if match and match[1] in tables:
                print('已存在，保留：'+match[1]); continue
            await conn.exec_driver_sql(statement)
            if match: print('已建表：'+match[1])


# 命令行与Web共用服务，避免出现两套不同的切分、双写或检索规则。
async def main(args):
    db=Database(DatabaseSettings().url()); service=build_service(db)
    try:
        if args.command=='init': await initialize(db)
        # 离线批量导入没有人工页面，但仍复用同一预览摘要校验和来源幂等逻辑。
        elif args.command=='import':
            materials=Materials(ROOT/'data'/'kb')
            for name in materials.names():
                p=materials.preview(name)
                print(name,await service.store.import_chunks(materials.checked(name,p['digest']),name))
        elif args.command=='vectorize':
            if args.crash_after_upsert:
                original=service.vectors.upsert
                async def crash(key,vector):
                    await original(key,vector)
                    print('故障注入：Milvus 已写入，MySQL 尚未回填。',flush=True)
                    # 刻意跳过finally模拟硬崩溃；仅用于恢复验收，不是正常关闭方式。
                    os._exit(73)
                service.vectors.upsert=crash
            print(json.dumps(await service.vectorize(),ensure_ascii=False))
        elif args.command=='mine': print(json.dumps(await build_miner(service).run(),ensure_ascii=False))
        elif args.command=='status': print(json.dumps(await service.store.stats(),ensure_ascii=False))
        elif args.command=='schedule':
            state_path=ROOT/'work'/'ch03'/'schedule.json'
            state_path.parent.mkdir(parents=True,exist_ok=True)
            print('调度已启动：北京时间每日02:00，错过的当日任务启动后补跑。',flush=True)
            while True:
                now=datetime.now(timezone.utc).astimezone(SHANGHAI)
                state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
                # 当天02:00已过且无完成标记时补跑；持续运行时也由这一分支触发每日任务。
                if now.hour>=2 and state.get('completed_date')!=now.date().isoformat():
                    try:
                        result=await build_miner(service).run()
                        # 调度顺序是挖知识后补向量，新增及合并问法的pending都会被处理。
                        vectors=await service.vectorize()
                        state={'completed_date':now.date().isoformat(),'mining':result,'vectors':vectors}
                        # 存在失败时不标记当天完成，让后续循环继续补跑。
                        if vectors['failed']: state.pop('completed_date')
                    except Exception as exc:
                        state={'error':type(exc).__name__,'attempt_at':now.isoformat()}
                    # 只在整条链结束后记录当天结果；失败仅记异常类型，不保存密钥或请求头。
                    temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(state_path)
                    print(json.dumps(state,ensure_ascii=False),flush=True)
                    if 'completed_date' not in state: await asyncio.sleep(60); continue
                # 短间隔便于正常关闭；时钟跨天后重新计算，不累积24小时漂移。
                await asyncio.sleep(min(30,max(1,(next_run()-datetime.now(timezone.utc)).total_seconds())))
    finally:
        # 正常退出释放向量连接和数据库引擎；故障注入的硬退出特意不走这里。
        await service.vectors.close(); await db.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['init','import','vectorize','mine','status','schedule'])
    parser.add_argument('--crash-after-upsert',action='store_true',help='仅验收用：写入第一条向量后退出73，不回填MySQL')
    args=parser.parse_args()
    try: asyncio.run(main(args))
    except KeyboardInterrupt: pass
