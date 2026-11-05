"""复用Ch04检索/生成/忠实度指标，逐题留报告，每个已启动轮次落一行。"""

import asyncio
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from sqlalchemy import select
from app.learning.models import EvalRun
from app.learning.confidence import assess
from app.rag.evaluation import retrieval_metrics, mean_or_none
from app.rag.quality import judge_answer
from app.rag.store import values

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def summarize(rows, expected):
    completed = [r for r in rows if 'error' not in r]
    known = [r for r in completed if r['relevant_ids']]
    unknown = [r for r in completed if not r['relevant_ids']]
    faith = [
        r['faith']['faithfulness']
        for r in completed
        if r['faith']['faithfulness'] is not None
    ]
    result = dict(
        total=expected,
        completed=len(completed),
        errors=len(rows) - len(completed),
        unprocessed=expected - len(rows),
        status='complete' if len(completed) == expected else 'partial',
        faithfulness=mean_or_none(faith),
        faithfulness_answered_n=len(faith),
        answer_rate=mean_or_none([r['outcome']['useful'] for r in completed]),
        answerable_answer_rate=mean_or_none([r['outcome']['useful'] for r in known]),
        unknown_refusal_rate=mean_or_none(
            [not r['outcome']['useful'] for r in unknown]
        ),
    )
    for name in ('recall@1', 'recall@5', 'recall@10', 'mrr@10'):
        result[name] = mean_or_none([r['metrics'][name] for r in known])
    return result


class EvaluationRunner:
    def __init__(
        self,
        db,
        knowledge,
        model,
        profile,
        *,
        dataset=None,
        reports=None,
        judge=None,
        model_name='configured',
    ):
        self.db, self.knowledge, self.model, self.profile = (
            db,
            knowledge,
            model,
            profile,
        )
        self.dataset = Path(dataset or ROOT / 'tests/fixtures/ch04_eval.json')
        self.reports = Path(reports or ROOT / 'dev-notes')
        self.judge, self.model_name = judge or judge_answer, model_name
        if not hasattr(db, 'evaluation_lock'):
            db.evaluation_lock = asyncio.Lock()
        self.progress = {}

    async def runs(self):
        async with self.db.sessions() as session:
            recent = [
                values(row)
                for row in await session.scalars(
                    select(EvalRun)
                    .order_by(EvalRun.created_at.desc(), EvalRun.id.desc())
                    .limit(1000)
                )
            ]
            # 图表按时间正序；先取最新窗口，重启调度才能读到真正最近的一轮。
            return list(reversed(recent))

    async def run(self, triggered_by='手动'):
        if triggered_by not in ('手动', '定时'):
            raise ValueError('评估触发方式无效')
        if self.db.evaluation_lock.locked():
            raise ValueError('已有评估轮次运行中')
        async with self.db.evaluation_lock:
            # 与本机原文写入共用原锁，整轮固定语料；这里不持有SQL事务。
            guard = (
                self.knowledge.store.writer()
                if hasattr(self.knowledge.store, 'writer')
                else nullcontext()
            )
            async with guard:
                return await self._run(triggered_by)

    async def _run(self, triggered_by):
        cases = json.loads(self.dataset.read_text(encoding='utf-8'))
        chunks = await self.knowledge.store.all_chunks()
        for case in cases:
            relevant = []
            for label in case['evidence']:
                matched = [
                    r['id']
                    for r in chunks
                    if (r.get('section_path') or '').endswith(' / ' + label)
                    or label.startswith('型号 ')
                    and label in r['questions']
                ]
                if not matched:
                    raise ValueError('冻结集标注缺少对应知识：' + case['id'])
                relevant.extend(matched)
            case['relevant_ids'] = sorted(set(relevant))
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / f'ch09-eval-{time.time_ns()}.json'
        version = dict(
            dataset_sha256=hashlib.sha256(self.dataset.read_bytes()).hexdigest(),
            corpus_sha256=digest(
                [
                    {
                        k: r.get(k)
                        for k in (
                            'id',
                            'category',
                            'questions',
                            'answer',
                            'content_type',
                            'section_path',
                        )
                    }
                    for r in chunks
                ]
            ),
            confidence_sha256=digest(self.profile),
            model=self.model_name,
            strategy='hybrid_rerank',
            pipeline='Ch04检索/结构化生成/忠实度裁判，Ch09证据闸；不代替多轮Agent回归',
        )
        report = dict(
            triggered_by=triggered_by,
            dataset=str(self.dataset),
            version=version,
            results=[],
            limitation='同一上游模型自动自评，拒答Faithfulness=N/A；部分失败不记成完整通过',
        )

        def save():
            self.progress = summarize(report['results'], len(cases))
            report['metrics'] = {
                **self.progress,
                'version': version,
                'report': str(path),
            }
            temp = path.with_suffix('.tmp')
            temp.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            temp.replace(path)

        save()
        cancelled = False
        try:
            for case in cases:
                row = dict(case)
                try:
                    retrieval = await self.knowledge.retriever.search(
                        case['query'], case.get('filters'), 'hybrid_rerank'
                    )
                    confidence = assess(retrieval['ranked'], self.profile)
                    retrieval = {
                        **retrieval,
                        'low_confidence': not confidence['allowed'],
                    }
                    row.update(
                        retrieval=retrieval,
                        confidence=confidence,
                        metrics=retrieval_metrics(
                            retrieval['ranked'], case['relevant_ids']
                        ),
                    )
                    row['outcome'] = await self.knowledge.generator.generate(
                        case['query'], retrieval
                    )
                    row['faith'] = await self.judge(
                        self.model, case['query'], row['outcome']
                    )
                except Exception as exc:
                    row['error'] = type(exc).__name__
                report['results'].append(row)
                save()
        except asyncio.CancelledError:
            cancelled = True
            report['cancelled'] = True
        finally:
            save()
            async with self.db.sessions.begin() as session:
                session.add(
                    EvalRun(
                        triggered_by=triggered_by,
                        dataset_size=len(cases),
                        metrics=report['metrics'],
                    )
                )
        if cancelled:
            raise asyncio.CancelledError
        return report['metrics']
