r"""看失败题的真实答案，判断是 Agent 错还是我判定错。
跑法：python 看失败.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

d = json.loads((Path(__file__).parent / "runs.json").read_text(encoding="utf-8"))

for cid in ["V2", "V3", "H2", "T2"]:
    c = d["cases"][cid]
    print("=" * 74)
    print(f"{cid} | {c['question']}")
    print(f"  期望工具 {c['expected_tools']} | 答案必含 {c['must_have']}")
    for i, r in enumerate(c["runs"][:2]):
        tag = "过" if r["passed"] else "错"
        print(f"  --run{i}[{tag}] 工具={r['tools_used']} 步数={r['steps']}")
        print(f"    答案：{r['answer'][:330]}")
