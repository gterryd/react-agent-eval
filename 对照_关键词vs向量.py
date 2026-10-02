r"""
★★ A/B 对照实验：关键词检索 vs 向量检索

靶子来自 8.30 的真实翻车：
  问「严格准确率 是多少」→ 关键词版把讲「显存要多少」的段落排第 1（"多少"刷分）

判定规则写在代码里，程序自己判对错，不由我口头说：
  每题给一个「正确答案必须包含的字符串」，看 Top-1 命中没有。

跑法：
  cd <项目目录>
  python 对照_关键词vs向量.py
"""
import sys

import kb
import vec

sys.stdout.reconfigure(encoding="utf-8")

# (问题, Top-1 片段里必须出现的字符串 —— 用来判定检索对不对)
CASES = [
    ("严格准确率 是多少", "0.3022"),
    ("交通锥 数不准 是什么原因", "交通锥"),
    ("42 辆车 是怎么发现的", "42"),
    ("LoRA 的 rank 和 alpha 设成多少", "16"),
    ("训练用了多长时间", "24987"),
]


def hit(chunks, must):
    """Top-1 是否命中。★只看第 1 名，这是最严的判法"""
    return bool(chunks) and must in chunks[0].text


if __name__ == "__main__":
    print(f"知识库：{len(kb.get_chunks())} 个片段\n")
    kw_ok = ve_ok = 0

    for q, must in CASES:
        kw = kb.search(q, top_k=1)
        ve = vec.search(q, top_k=1, with_score=True)
        ve_chunks = [c for _, c in ve]
        a, b = hit(kw, must), hit(ve_chunks, must)
        kw_ok += a
        ve_ok += b

        print(f"{'='*72}\n问：{q}   （Top-1 里必须出现 {must!r}）")
        print(f"  关键词 {'✓' if a else '✗'}  →  "
              f"{kw[0].heading if kw else '（无命中）'}")
        print(f"  向量   {'✓' if b else '✗'}  →  "
              f"{ve_chunks[0].heading}   相似度={ve[0][0]:.3f}")
        if a != b:
            print(f"  ★ 两者结论不同")

    n = len(CASES)
    print(f"\n{'='*72}\n★ Top-1 准确率：关键词 {kw_ok}/{n} = {kw_ok/n*100:.0f}%   |   "
          f"向量 {ve_ok}/{n} = {ve_ok/n*100:.0f}%")
    print("★ 以上每个数字都是程序当场跑出来的，没有写死")
