"""共享向量化和检索服务：页面、离线入口和聊天共用。"""
import asyncio
import math
from app.knowledge.splitter import embedding_text


# 拒绝空向量、NaN/无穷大和零向量，避免无意义的相似度检索。
def validate_vector(vector):
    if not vector or not all(math.isfinite(v) for v in vector) or sum(v*v for v in vector)==0:
        raise ValueError('嵌入向量无效')


class KnowledgeService:
    def __init__(self, store, embeddings, vectors, threshold=.5, top_k=5):
        # 注入原文仓库、嵌入模型和向量库；单测可替换为不联网的假实现。
        self.store,self.embeddings,self.vectors=store,embeddings,vectors
        self.threshold,self.top_k=threshold,top_k

    async def vectorize(self, stop=None, progress=None):
        """逐块提交；未知写结果仍 pending，重跑同主键 upsert 修复。"""
        result={'succeeded':0,'failed':0,'errors':[]}
        async with self.store.writer():
            # 恢复依据是数据库 pending，而不是上次任务处理到了第几条。
            rows=[r for r in await self.store.all_chunks() if r['vectorize_status']=='pending']
            result['total']=len(rows)
            for row in rows:
                if stop and stop.is_set(): break
                try:
                    # 只拼分类、问法、答案；6000是字符安全预算，超长原文仍留待处理。
                    text=embedding_text(row)
                    if len(text)>6000: raise ValueError('内容超过安全输入预算')
                    vector=(await self.embeddings.aembed_documents([text]))[0]
                    validate_vector(vector)
                    # 模型请求完成后再检查停止信号；尚未写向量的块继续保持 pending。
                    if stop and stop.is_set(): break
                    # MySQL 和 Milvus 没有跨库事务，因此使用稳定主键 upsert + pending 重试。
                    await self.vectors.upsert(str(row['id']),vector)
                    # 此处进程中断，pending 保留；同主键重跑不会产生第二条。
                    await self.store.mark_done(row['id'])
                    result['succeeded']+=1
                # 单块失败不阻断后续块；只记录异常类型，避免把敏感上游请求信息暴露到页面。
                except Exception as exc:
                    result['failed']+=1
                    result['errors'].append({'id':row['id'],'error':type(exc).__name__})
                if progress: progress(dict(result))
                # 主动交还事件循环，让状态查询和停止请求有机会执行。
                await asyncio.sleep(0)
        result['stopped']=bool(stop and stop.is_set())
        return result

    async def search(self, query, diagnostics=False):
        """分数只作 dense 阈值过滤；正文必须回到 MySQL 且状态为 done。"""
        vector=await self.embeddings.aembed_query(query)
        validate_vector(vector)
        # 先由 Milvus 取Top-K，再过滤低分和非done记录；最终返回可能少于K条。
        hits=await self.vectors.search(vector,self.top_k)
        # 正文以MySQL为准；Milvus里即使残留一条向量，也不能绕过pending状态。
        rows={str(r['id']):r for r in await self.store.all_chunks()}
        result=[]
        for key,score in hits:
            row=rows.get(str(key))
            if row is None or row['vectorize_status']!='done' or score<self.threshold: continue
            # 对外继续使用单数 question 字段，保持旧 query_faq 出参契约。
            item={k:row[k] for k in ('id','answer','category')}; item['question']=row['questions']
            # 仅管理页需要分数与来源；聊天工具仍只返回原来的四个字段。
            if diagnostics: item.update(score=score,section_path=row['section_path'],content_type=row['content_type'])
            result.append(item)
        return result
