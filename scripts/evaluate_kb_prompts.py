"""对抽取及去重 Prompt 运行标注评估；不会把评估问答写业务表。"""
import asyncio,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from app.core.llm import get_chat_model
from app.knowledge.mining import Extracted,Decision,EXTRACT_PROMPT,DEDUPE_PROMPT,safe_qa,structured


# 直接评估真实模型的抽取/去重判断，不写业务数据库，也不运行整条定时链。
async def main():
    fixtures=json.loads((ROOT/'tests/fixtures/kb_prompt_cases.json').read_text(encoding='utf-8'))
    model=get_chat_model()
    sources=[{k:v for k,v in r.items() if k!='keep'} for r in fixtures['extraction']]
    extracted=await structured(model,Extracted,EXTRACT_PROMPT,sources)
    # 抽取结果先经过与生产相同的出处和敏感片段检查，再与保留标注比较。
    accepted=[q.model_dump() for q in extracted.items if safe_qa(q.model_dump(),sources)]
    results=[{'source_ref':r['source_ref'],'passed':any(q['source_ref']==r['source_ref'] for q in accepted)==r['keep']} for r in fixtures['extraction']]
    # 去重用例核对new/duplicate/conflict；每例异常也作为失败保存在报告。
    for case in fixtures['dedupe']:
        try:
            answer=await structured(model,Decision,DEDUPE_PROMPT,{k:v for k,v in case.items() if k!='expected'})
            results.append({'expected':case['expected'],'actual':answer.model_dump(),'passed':answer.decision==case['expected']})
        except Exception as exc:
            results.append({'expected':case['expected'],'error':type(exc).__name__,'passed':False})
    report={'passed':sum(r['passed'] for r in results),'total':len(results),'extraction':extracted.model_dump(),'results':results}
    path=ROOT/'dev-notes'/f'ch03-prompts-{time.time_ns()}.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    print(path)


if __name__=='__main__': asyncio.run(main())
