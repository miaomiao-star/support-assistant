from app.knowledge.splitter import split_markdown, embedding_text


def test_headings_fences_and_metadata():
    """验证标题栈、围栏内伪标题，以及元数据不进入向量文本。"""
    rows = split_markdown('# 政策\n## 退货\n第一句。\n```text\n# 不是标题\n```\n### 条件\n完整商品。', 'a.md', 'policy')
    assert [r.questions for r in rows] == ['退货', '条件']
    assert rows[1].category == '政策 / 退货'
    assert '# 不是标题' in rows[0].answer
    assert 'a.md' not in embedding_text(rows[0])
    assert 'policy' not in embedding_text(rows[0])


def test_sentence_overlap_never_loses_tail():
    """用小字符预算逼出多块，检查完整句重叠和末尾正文没有遗漏。"""
    text = '甲甲甲。乙乙乙。丙丙丙。丁丁丁。戊戊戊。'
    rows = split_markdown('# 标题\n'+text, 'x.md', 'manual', size=14, overlap=4)
    assert len(rows) > 1
    assert all(r.answer.endswith('。') for r in rows)
    assert rows[1].answer.startswith('丙丙丙。')
    assert '戊戊戊。' in rows[-1].answer


def test_table_headers_repeated_and_all_rows_preserved():
    """每块重复表头，每条数据行恰好保留一次。"""
    head='| 问题 | 时限 |\n| --- | --- |'
    body=[f'| 条目{i} | 3天 |' for i in range(12)]
    rows=split_markdown('# 表\n'+head+'\n'+'\n'.join(body),'t.md','manual',size=80)
    assert len(rows)>1
    assert all(r.answer.startswith(head) for r in rows)
    assert all(sum(line in r.answer for r in rows)==1 for line in body)


def test_unbroken_sentence_preserved_and_marked():
    """不可安全拆开的长句必须保留原文并提示超长。"""
    text='长'*100+'。'
    rows=split_markdown('# 长句\n'+text,'l.md','manual',size=20)
    assert len(rows)==1 and rows[0].answer==text and rows[0].oversized


def test_real_documents_keep_each_faq_question():
    """使用真实材料检查标题问法与运费原文，不调用嵌入模型。"""
    from pathlib import Path
    rows=split_markdown(Path('data/kb/product-faq.md').read_text(encoding='utf-8'),'product-faq.md','faq')
    assert any(r.questions=='运费怎么算' and '10 元' in r.answer for r in rows)
    assert all(r.section_path.startswith('product-faq.md / ') for r in rows)
