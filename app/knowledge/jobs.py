"""页面发起的长任务：显式停止、可刷新状态、进程重启后不伪装仍运行。"""
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path


class Jobs:
    def __init__(self,root):
        self.path=Path(root)/'job.json'
        self.task=None; self.event=asyncio.Event()
        self.info=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'state':'idle'}
        # 重启后内存里的任务已不存在，不能继续向页面报告“运行中”。
        if self.info['state'] in ('running','stopping'):
            self.info.update(state='interrupted',message='服务已重启，可补齐待处理知识')
            self.save()

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        # 先写临时文件再替换，降低进程中断留下半截JSON的风险。
        temp=self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.info,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(self.path)

    def status(self): return dict(self.info)

    # 这里只创建后台协程并立即返回状态，不让HTTP请求等待整个建库任务。
    def start(self,kind,worker):
        if self.task and not self.task.done(): raise ValueError('已有任务运行中')
        # 每个新任务重新创建停止事件，避免沿用上一次的停止信号。
        self.event=asyncio.Event()
        self.info={'kind':kind,'state':'running','started_at':datetime.now(timezone.utc).isoformat(),'result':{}}
        self.save()
        # 业务服务每完成一项就回调保存；页面轮询读取这份任务快照。
        def progress(result): self.info['result']=result; self.save()
        async def run():
            try:
                result=await worker(self.event,progress)
                self.info.pop('message',None)
                self.info.update(result=result,state='stopped' if result.get('stopped') else ('partial' if result.get('failed') else 'done'))
            # 服务关闭触发取消属于中断，保留已提交结果并继续传播取消。
            except asyncio.CancelledError:
                self.info.update(state='interrupted',message='任务中断，已提交结果保留，可补跑')
                raise
            except Exception as exc:
                self.info.update(state='error',message='任务未完成：'+type(exc).__name__)
            finally: self.save()
        self.task=asyncio.create_task(run())
        return self.status()

    # 用户停止只设信号；让业务代码在操作边界退出，不强行终止数据库提交。
    def stop(self):
        if self.task and not self.task.done():
            self.event.set(); self.info.update(state='stopping',message='正在结束当前操作，已完成结果保留'); self.save()
        return self.status()

    # 应用退出时等待任务结束；向量线程封装会处理底层同步写入尚未结束的情况。
    async def close(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try: await self.task
            except asyncio.CancelledError: pass
