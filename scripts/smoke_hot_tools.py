"""运行中的客服热注册验收；不修改图、不重启客服，结果落真实审计。"""

import argparse
import json
import sys
import time
from uuid import uuid4
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import httpx
from sqlalchemy import select
from app.db import Database, DatabaseSettings
from app.models import ToolAuditLog
from app.workflow.persistence import run_async
from scripts.smoke_tools import frames


async def main(url, pid, remote_tool, remote_question, remote_marker):
    # 调用前应已投放注册文件/重启指定MCP并热更本地权限；此脚本不替用户改运行环境。
    # pid只记录调用方提供的前值，是否同进程还需独立核对后值，不能单靠此字段证明。
    report = {'server_pid_before': pid, 'results': []}
    path = ROOT / 'dev-notes' / f'ch08-hot-{time.time_ns()}.json'
    db = Database(DatabaseSettings().url())
    try:
        async with httpx.AsyncClient(
            base_url=url,
            trust_env=False,
            timeout=180,
            headers={'X-Visitor-Id': 'hot-' + uuid4().hex},
        ) as client:
            for name, question, marker in [
                (
                    'query_demo_service_code',
                    '请调用工具查询本系统的演示服务识别码是什么',
                    'MEW-08',
                ),
                (
                    remote_tool,
                    remote_question,
                    remote_marker,
                ),
            ]:
                row = {'name': name, 'question': question}
                try:
                    response = await client.post(
                        '/api/chat',
                        json=dict(session_id='hot-' + uuid4().hex, message=question),
                    )
                    events = frames(response.text)
                    assert not any(k == 'error' for k, v in events), events
                    cid = int(next(v['id'] for k, v in events if k == 'conversation'))
                    answer = ''.join(
                        v.get('delta', '')
                        for k, v in events
                        if k == 'message' and isinstance(v, dict)
                    )
                    async with db.sessions() as session:
                        # 以真实审计确认Agent实际执行过工具，不能只凭回答里出现识别码。
                        rows = (
                            await session.scalars(
                                select(ToolAuditLog).where(
                                    ToolAuditLog.conversation_id == cid,
                                    ToolAuditLog.tool_name == name,
                                )
                            )
                        ).all()
                    assert rows and rows[-1].status == '成功' and marker in answer
                    row.update(
                        passed=True,
                        cid=cid,
                        answer=answer,
                        source=rows[-1].tool_source,
                        server=rows[-1].mcp_server,
                        status=rows[-1].status,
                    )
                except Exception as exc:
                    row.update(
                        passed=False, error=type(exc).__name__, detail=str(exc)[:400]
                    )
                report['results'].append(row)
                path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
                )
                print(('PASS' if row['passed'] else 'FAIL') + ' ' + name, flush=True)
    finally:
        await db.close()
    report.update(
        passed=sum(r['passed'] for r in report['results']), total=len(report['results'])
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{report['passed']}/{report['total']} {path}")
    return report['passed'] == report['total']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--remote-tool', default='query_carrier_hotline')
    parser.add_argument(
        '--remote-question', default='请查询订单1001的物流承运商模拟客服电话'
    )
    parser.add_argument('--remote-marker', default='400-000-0008')
    args = parser.parse_args()
    raise SystemExit(
        0
        if run_async(
            main(
                args.url,
                args.server_pid,
                args.remote_tool,
                args.remote_question,
                args.remote_marker,
            )
        )
        else 1
    )
