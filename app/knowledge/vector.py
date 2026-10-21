"""官方 PyMilvus 同步客户端封装；线程操作结束前不并发重连。"""
import asyncio
import threading
from pymilvus import MilvusClient, DataType


class MilvusVectors:
    def __init__(self, uri, collection='knowledge'):
        self.uri,self.collection=uri,collection
        self.client=None
        # 同步客户端在线程中使用，线程锁保护连接初始化、读写和关闭。
        self.lock=threading.Lock()

    async def _call(self, work):
        """取消协程不能停止线程；等待未知写结果收敛后才释放外层写锁。"""
        # 同步SDK移到工作线程执行，避免阻塞FastAPI的事件循环。
        task=asyncio.create_task(asyncio.to_thread(work))
        try:
            # shield阻止外层取消直接取消等待对象；底层同步线程并不会因此停止。
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try: await asyncio.shield(task)
            except Exception: pass
            raise

    # 首次写入可建集合；纯查询发现集合不存在时返回空，不隐式建库。
    def _ready(self, dim, create=False):
        if self.client is None: self.client=MilvusClient(uri=self.uri,timeout=10)
        if not self.client.has_collection(self.collection):
            if not create: return False
            self.client.create_collection(collection_name=self.collection,dimension=dim,
                id_type='string',max_length=64,auto_id=False,metric_type='COSINE')
        # 检查已有集合的维度、主键和度量；不匹配就报错，不自动删库重建。
        info=self.client.describe_collection(self.collection)
        fields={f['name']:f for f in info['fields']}
        if int(fields['vector']['params']['dim'])!=dim or fields['id']['type']!=DataType.VARCHAR or info['auto_id']:
            raise ValueError('已有向量集合 schema 不匹配，未执行重建')
        index=self.client.describe_index(self.collection,'vector')
        if index.get('metric_type')!='COSINE': raise ValueError('已有集合度量不是COSINE，未执行重建')
        # 服务重启后已有集合仍需加载，才能稳定恢复查询。
        self.client.load_collection(self.collection)
        return True

    # 同一MySQL id始终对应同一Milvus主键，重试更新原记录。
    async def upsert(self,key,vector):
        def work():
            with self.lock:
                self._ready(len(vector),True)
                self.client.upsert(collection_name=self.collection,data=[{'id':key,'vector':vector}])
        await self._call(work)

    async def search(self,vector,limit):
        def work():
            with self.lock:
                if not self._ready(len(vector)): return []
                hits=self.client.search(collection_name=self.collection,data=[vector],limit=limit,
                    output_fields=['id'],search_params={'metric_type':'COSINE'})
                # COSINE下SDK的distance表示相似度，越大越相似；上层使用它过滤阈值。
                return [(str(h['id']),float(h['distance'])) for h in hits[0]]
        return await self._call(work)

    async def close(self):
        def work():
            with self.lock:
                if self.client: self.client.close(); self.client=None
        await self._call(work)
