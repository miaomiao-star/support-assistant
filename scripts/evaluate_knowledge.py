"""真实嵌入/检索评估；真值先写入 fixtures，报告每次单独保存。"""
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parent.parent


# 调用真实检索接口，衡量召回质量；不会执行聊天模型的最终回答生成。
def main():
    cases=json.loads((ROOT/'tests/fixtures/kb_cases.json').read_text(encoding='utf-8'))
    results=[]
    for case in cases:
        started=time.monotonic()
        r=requests.post('http://127.0.0.1:8000/api/kb/search',json={'query':case['query']},timeout=90)
        r.raise_for_status(); hits=r.json()['data']
        # 正例要求Top-K包含标注章节；负例要求阈值过滤后没有任何命中。
        passed=not hits if case['expected'] is None else any(case['expected'] in h['question'] for h in hits)
        results.append({**case,'passed':passed,'seconds':round(time.monotonic()-started,3),'hits':hits})
        print(case['query'],passed,[(h['question'],round(h['score'],3)) for h in hits],flush=True)
    report={'passed':sum(r['passed'] for r in results),'total':len(results),'results':results}
    # 每次报告使用新时间戳，保留失败与修复后的历史，便于对比。
    path=ROOT/'dev-notes'/f'ch03-retrieval-{time.time_ns()}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path,report['passed'], '/', report['total'])


if __name__=='__main__': main()
