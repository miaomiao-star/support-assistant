"""知识原文仓库。外部网络调用不得放在数据库事务内部。"""
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import select
from app.knowledge.models import KnowledgeChunk, QAStaging


def record(row):
    # 只输出普通字段值，避免让已关闭 Session 的 ORM 对象流向业务层。
    return {c.name:getattr(row,c.name) for c in row.__table__.columns if c.name not in ('created_at','updated_at')}


class KnowledgeStore:
    def __init__(self, db):
        self.db=db
        # 共享同一 Database 的仓库复用协程锁，防止同进程任务互相穿插。
        if not hasattr(db,'knowledge_lock'): db.knowledge_lock=asyncio.Lock()

    @asynccontextmanager
    async def writer(self):
        """本机所有 Web/定时任务共享文件锁；测试数据库使用独立实例锁。"""
        if self.db.knowledge_lock.locked(): raise ValueError('已有知识写入任务，请稍后重试')
        async with self.db.knowledge_lock:
            handle=None
            try:
                # MySQL 路径另加文件锁，协调独立的 Web、命令行和调度进程。
                # 锁只覆盖本机；不等价于多服务器分布式锁。
                if self.db.engine.dialect.name=='mysql':
                    from pathlib import Path
                    root=Path(__file__).resolve().parents[2]/'work'/'ch03'
                    root.mkdir(parents=True,exist_ok=True)
                    # 准备至少一个字节供 Windows 锁定；锁随文件句柄关闭而释放。
                    handle=(root/'writer.lock').open('a+b'); handle.seek(0)
                    if handle.read(1)==b'': handle.write(b'0'); handle.flush()
                    handle.seek(0)
                    try:
                        import os
                        if os.name=='nt':
                            import msvcrt
                            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                        else:
                            import fcntl
                            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except OSError:
                        raise ValueError('已有离线任务正在写入知识库') from None
                # 调用方的整段写任务都在锁内，但外部模型调用不占数据库事务。
                yield
            finally:
                if handle: handle.close()

    # 教学规模直接读取全部知识；数据扩大后可改为按主键批量回查。
    async def all_chunks(self):
        async with self.db.sessions() as s:
            return [record(r) for r in await s.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))]

    async def stats(self):
        rows=await self.all_chunks()
        return {'total':len(rows),'pending':sum(r['vectorize_status']=='pending' for r in rows),
                'done':sum(r['vectorize_status']=='done' for r in rows)}

    async def import_chunks(self, chunks, source):
        """按完整来源块比较；同源变更拒绝，重复提交直接复用原主键。"""
        async with self.writer():
            async with self.db.sessions.begin() as s:
                # 用来源路径前缀识别同一批材料；autoescape 避免文件名中的通配符被当作查询语法。
                existing=list(await s.scalars(select(KnowledgeChunk).where(
                    KnowledgeChunk.section_path.startswith(source+' / ',autoescape=True)).order_by(KnowledgeChunk.id)))
                values=[c.record() for c in chunks]
                if existing:
                    old=[{k:getattr(r,k) for k in values[0]} for r in existing] if values else []
                    # 挖知识可以为原块追加真实问法；不应因此把原材料重放误判为改版。
                    same=len(old)==len(values) and all(
                        all(a[k]==b[k] for k in b if k!='questions') and
                        set(b['questions'].splitlines()).issubset(a['questions'].splitlines())
                        for a,b in zip(old,values))
                    if not same: raise ValueError('此来源已入库且内容变化；本章不支持覆盖，请使用新的来源版本')
                    return {'created':0,'existing':len(existing),'ids':[r.id for r in existing]}
                # 新块默认 pending；flush 先取得主键，再把同来源前后块一起连好。
                rows=[KnowledgeChunk(**v) for v in values]
                s.add_all(rows); await s.flush()
                for i,row in enumerate(rows):
                    row.prev_chunk_id=rows[i-1].id if i else None
                    row.next_chunk_id=rows[i+1].id if i+1<len(rows) else None
                return {'created':len(rows),'existing':0,'ids':[r.id for r in rows]}

    # 独立短事务回填成功状态；调用者必须先完成 Milvus upsert。
    async def mark_done(self, key):
        async with self.db.sessions.begin() as s:
            row=await s.get(KnowledgeChunk,key)
            if row is None: raise ValueError('知识块已不存在')
            row.vector_id=str(key); row.vectorize_status='done'

    # 管理页只取最近200条暂存结果，避免一次返回无限历史。
    async def staging(self):
        async with self.db.sessions() as s:
            return [record(r) for r in await s.scalars(select(QAStaging).order_by(QAStaging.id.desc()).limit(200))]
