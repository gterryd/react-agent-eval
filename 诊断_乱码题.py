r"""诊断：问「4bit 乱码」时，讲乱码根因的段落排第几名？
先看真实排名再决定要不要调，别瞎改。"""
import sys

import kb
import vec

sys.stdout.reconfigure(encoding="utf-8")

# 真答案的标志词（出自 技术复盘记录.md 真实原文）
TRUTH = ["数值不稳", "bitsandbytes", "误差累积", "纯 base"]

for q in ["4bit 乱码", "4bit 推理为什么输出乱码", "推理输出一堆感叹号 怎么排查的"]:
    print(f"\n{'='*70}\n问：{q}")
    for rank, (s, c) in enumerate(vec.search(q, top_k=8, with_score=True), 1):
        mark = "★真答案" if any(t in c.text for t in TRUTH) else ""
        print(f"  {rank}. {s:.3f}  {c.heading[:40]:<42}{mark}")
