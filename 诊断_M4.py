r"""诊断 M4：轿车23.9%/交通锥15.4% 明明在知识库里，为什么它说「查不到」？
跑法：python 诊断_M4.py
"""
import sys

import kb
import tools
import vec

sys.stdout.reconfigure(encoding="utf-8")

# 1) 这两个数字到底在哪几个片段里？
print("★ 步骤1：含『23.9』或『15.4』的片段（知识库里到底有没有）")
hit = [c for c in kb.get_chunks() if "23.9" in c.text or "15.4" in c.text]
for c in hit:
    print(f"  [{c.source} → {c.heading}]")
    print(f"    ...{c.text[max(0, c.text.find('23.9')-60):c.text.find('23.9')+80]}...")
print(f"  共 {len(hit)} 片\n")

# 2) 用它实际发的查询，看这些片段排第几
print("★ 步骤2：向量检索排名（Top-8），看真答案排第几")
for q in ["轿车 交通锥 准确率", "轿车和交通锥数不准 准确率是多少",
          "分类别 误差画像 准确率"]:
    print(f"\n  查询：{q}")
    for rank, (s, c) in enumerate(vec.search(q, top_k=8, with_score=True), 1):
        mark = " ★真答案" if ("23.9" in c.text or "15.4" in c.text) else ""
        print(f"    {rank}. {s:.3f}  {c.source[:14]:<16}{c.heading[:34]:<36}{mark}")

# 3) search_docs 实际会返回什么（含 0.35 相似度阈值过滤）
print("\n★ 步骤3：search_docs 实际返回（★注意 0.35 阈值可能把真答案滤掉）")
out = tools.search_docs("轿车 交通锥 准确率")
print(f"  返回长度 {len(out)} 字；含23.9={'23.9' in out}；含15.4={'15.4' in out}")
print(f"  前 300 字：{out[:300]}")
