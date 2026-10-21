"""材料范围、预览版本与手工知识规范化，不执行模型调用。"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from app.knowledge.splitter import Chunk, split_markdown, split_body


class Materials:
    def __init__(self,root):
        self.root=Path(root).resolve()

    # 只接受材料目录内的直接子文件，拒绝越级路径和指向其他目录的文件。
    def names(self):
        return sorted(p.name for p in self.root.glob('*.md') if p.is_file() and p.resolve().parent==self.root)

    def preview(self,name):
        if name not in self.names(): raise ValueError('请选择材料清单内的 Markdown 文件')
        path=self.root/name
        text=path.read_text(encoding='utf-8-sig')
        config_path=self.root.parent/'kb-manifest.json'
        config=json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
        # 清单显式标注类型与关键章节；未标注材料按手册处理。
        meta=config.get(name,{'type':'manual','key_sections':[]})
        # 摘要同时覆盖原文和标注，防止预览后任一内容变化却仍录入旧版本。
        digest=hashlib.sha256((text+json.dumps(meta,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
        rows=split_markdown(text,name,meta['type'],key_sections=meta.get('key_sections',[]))
        return {'name':name,'digest':digest,'chunks':[asdict(r) for r in rows]}

    # 录入前重新计算预览版本；预览本身不写数据库或调用模型。
    def checked(self,name,digest):
        result=self.preview(name)
        if result['digest']!=digest: raise ValueError('材料或标注已变化，请重新预览')
        return [Chunk(**r) for r in result['chunks']]


def manual_chunks(values):
    """手工来源由完整内容决定；重复点保存复用同一个来源。"""
    # 完整表单内容决定来源身份；同内容重复保存会复用已存在的知识主键。
    payload={k:values.get(k) for k in ('category','questions','answer','section_path','content_type','is_key_clause')}
    source='manual:'+hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    path=source+' / '+(values.get('section_path') or '手工录入')
    if len(path)>512: raise ValueError('手工章节路径过长')
    # 长答案可拆多块，每块沿用同一分类、真实问法和人工标注。
    chunks=[Chunk(values['category'],values['questions'],answer,path,values.get('content_type','faq'),
                  bool(values.get('is_key_clause')),len(answer)>800) for answer in split_body(values['answer'],800,120)]
    return chunks,source
