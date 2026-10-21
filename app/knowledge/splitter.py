"""保留结构的 Markdown 切分；标题只提供上下文，不改写商家的原文。"""
import re
from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    # 前三项是知识语义；后面的字段是追溯、预览用元数据。
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool = False
    # 只提示人工检查，不存数据库；不会为凑长度截掉半句话。
    oversized: bool = False

    def record(self):
        """预览警告不是数据库字段。"""
        return {k: v for k, v in asdict(self).items() if k != 'oversized'}


def embedding_text(chunk):
    """只有三格进入 embedding；来源、类型、指针和关键标志均排除。"""
    get = chunk.get if isinstance(chunk, dict) else lambda k: getattr(chunk, k)
    return '\n'.join(f'{label}：{get(k)}' for label, k in
                     [('分类', 'category'), ('问法', 'questions'), ('答案', 'answer')])


def sentences(text):
    """只在句末/结构换行处分隔，不在数字的小数点或词中间切断。"""
    return [p for p in re.split(r'(?<=[。！？!?])|(?<=\.)\s+|\n', text) if p.strip()]


def split_body(body, size, overlap):
    """先识别不可拆结构，再按段落、完整句子递归缩小。"""
    # 这里的分层拆解通过顺序扫描实现，没有递归调用；优先保留结构，再按句子装块。
    lines = body.strip().splitlines()
    # 第一遍把正文拆成结构单元；长度按字符计，不是模型 token 数。
    units, i = [], 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^\s*(`{3,}|~{3,})', line):
            # 代码围栏整体保留，避免把代码中的换行或标点当成普通句子。
            marker = re.match(r'^\s*(`{3,}|~{3,})', line)[1]
            code = [line]; i += 1
            while i < len(lines):
                code.append(lines[i]); i += 1
                if re.match(r'^\s*'+re.escape(marker[0])+r'{'+str(len(marker))+r',}\s*$', code[-1]):
                    break
            units.append(('block', '\n'.join(code)))
        elif i+1 < len(lines) and '|' in line and re.match(r'^\s*\|?\s*:?-{3,}', lines[i+1]):
            # 表头含列名和 Markdown 分隔行；每次切换表格块都重新复制。
            header = line+'\n'+lines[i+1]; i += 2
            current = header
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                if current != header and len(current)+len(lines[i])+1 > size:
                    units.append(('table', current)); current = header
                current += '\n'+lines[i]; i += 1
            units.append(('table', current))
        elif line.strip():
            units.extend(('sentence', p.strip()) for p in sentences(line)); i += 1
        else:
            i += 1
    # 第二遍将单元装入目标大小的块；不可拆单元允许超出目标长度。
    out, current = [], []
    for kind, value in units:
        # 表格独立成块，避免普通句子的重叠逻辑重复表格行。
        if kind == 'table':
            if current: out.append('\n'.join(current)); current = []
            out.append(value); continue
        if current and len('\n'.join(current+[value])) > size:
            out.append('\n'.join(current))
            tail, count = [], 0
            # 重叠只选句末完整的末尾片段；代码块不参与重复。
            for item in reversed(current):
                if not re.search(r'[。！？!?\.]$', item) or count+len(item)>overlap: break
                tail.insert(0,item); count += len(item)
            current = tail
            # 重叠挤占新句子空间时舍弃重叠，保留正文完整性。
            if len('\n'.join(current+[value])) > size: current = []
        current.append(value)
    if current: out.append('\n'.join(current))
    return out


def split_markdown(text, source, content_type, size=800, overlap=120, key_sections=()):
    """维护标题栈；遇到标题才结束章节，代码围栏内的 # 只是正文。"""
    if size < 8 or overlap < 0: raise ValueError('切块参数无效')
    # 标题栈保存当前祖先路径；body 收集当前章节正文，fence 标记代码围栏。
    stack, body, sections, fence = [], [], [], None
    def finish():
        if any(s.strip() for s in body): sections.append(([t for _,t in stack], '\n'.join(body)))
        body.clear()
    for line in text.splitlines():
        marker = re.match(r'^\s*(`{3,}|~{3,})',line)
        if marker:
            token=marker[1]
            if fence is None: fence=token
            elif token[0]==fence[0] and len(token)>=len(fence): fence=None
            body.append(line); continue
        heading = re.match(r'^ {0,3}(#{1,6})\s+(.+?)\s*#*$',line) if fence is None else None
        if heading:
            finish(); level=len(heading[1])
            # 同级/更浅标题到来时弹出旧分支，留下真正的上级标题。
            stack=[(n,t) for n,t in stack if n<level]+[(level,heading[2])]
        else: body.append(line)
    finish()
    chunks=[]
    for headings, body in sections:
        # FAQ 标题使用真实问法；政策/手册以章节标题代替问题。
        title=headings[-1] if headings else source
        path=' / '.join([source]+headings)
        if len(path)>512: raise ValueError('章节路径超过512字符')
        # category 不带文件名；section_path 另存文件名与完整标题路径。
        category=' / '.join(headings[:-1]) or (headings[0] if headings else source)
        if len(category)>255: raise ValueError('分类超过255字符')
        for answer in split_body(body,size,overlap):
            chunks.append(Chunk(category,title,answer,path,content_type,
                                title in key_sections,len(answer)>size))
    return chunks
