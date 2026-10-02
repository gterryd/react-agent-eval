r"""第6步-② 全链路日志：查看工具

★这一步解决的问题：Agent 答完题，过程就没了 —— 排查时不知道它中间干了什么。
  现在每次问答都落盘：第几步想了什么、调了哪个工具、传什么参数、查到什么、耗时多少、烧了多少 token。
  ★这就是 JD 里的「Agent 可观测 / 全链路追踪」。

跑法（cmd，先 cd 到项目目录）：
  python 看日志.py          ← 列出全部
  python 看日志.py 最后      ← 看最后一次的完整现场
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path(__file__).parent / "logs"
INDEX = LOG_DIR / "index.jsonl"


COLD_START_HINT = (
    "⚠️ 第一次调用 search_docs 会顺带加载 BGE 模型（约15s，一次性）。\n"
    "   这不是检索慢 —— 后续调用只要 0.008s 左右。\n"
    "   ★统计耗时时必须把冷启动那次单独标出来，否则会得出「63%时间花在检索」这种错结论。"
)


def load_index():
    if not INDEX.exists():
        print("还没有日志。先跑一次 看溯源.py 或 问一句.py")
        sys.exit(0)
    return [json.loads(l) for l in INDEX.read_text(encoding="utf-8").splitlines() if l.strip()]


def 列表():
    rows = load_index()
    print(f"共 {len(rows)} 次运行\n")
    print(f"{'时间':<20}{'步':<4}{'总耗时':<8}{'等模型':<8}{'跑工具':<8}{'token':<8}问题")
    for r in rows:
        flag = " ←冷启动(含加载模型)" if r["工具耗时合计"] >= 3.0 else ""
        print(f"{r['started_at'][5:]:<20}{r['steps']:<4}{r['总耗时']:<8}"
              f"{r['llm耗时合计']:<8}{r['工具耗时合计']:<8}{r['total_tokens']:<8}"
              f"{r['question'][:22]}{flag}")
    # ★汇总：这几个数字就是 README 里的工程指标
    n = len(rows)
    lat = sorted(r["总耗时"] for r in rows)
    print(f"\n★ 汇总 {n} 次：")
    print(f"  平均步数 {sum(r['steps'] for r in rows)/n:.2f}")
    print(f"  平均 token {sum(r['total_tokens'] for r in rows)/n:.0f}")
    print(f"  延迟 中位 {lat[n//2]:.2f}s / P95 {lat[max(0, int(n*0.95)-1)]:.2f}s")
    # ★★耗时拆分（能回答「该优化哪边」：等模型=API没法优化；跑工具=检索要优化）
    #   ⚠️ 必须剔掉冷启动：首次调 search_docs 会加载 BGE 模型（约15s，一次性）。
    #   9.24 教训：不剔就会算出「63% 时间花在检索上」—— 完全相反的错结论。
    COLD = 3.0   # 单次工具耗时超过 3s，只可能是模型加载，不是检索
    warm = [r for r in rows if r["工具耗时合计"] < COLD]
    cold = [r for r in rows if r["工具耗时合计"] >= COLD]
    if cold:
        cold_str = "/".join(f"{r['工具耗时合计']:.1f}s" for r in cold)
        print(f"\n  ⚠️ 剔除 {len(cold)} 次冷启动（工具耗时 {cold_str}"
              f"，那是加载 BGE 模型、不是检索）")
    if warm:
        tl = sum(r["llm耗时合计"] for r in warm)
        tt = sum(r["工具耗时合计"] for r in warm)
        sh = tl / (tl + tt) * 100 if (tl + tt) else 0
        print(f"  ★耗时去哪了（{len(warm)} 次热态）：等模型 {tl:.2f}s（{sh:.1f}%）| "
              f"跑工具 {tt:.3f}s（{100-sh:.1f}%）")
        print(f"  → 结论：**瓶颈在 API 延迟，不在检索**。检索已快到可忽略。")


def 详情(run_id=None):
    """★看一次运行的完整决策链：想了什么 → 做了什么 → 拿到什么"""
    rows = load_index()
    rid = run_id or rows[-1]["run_id"]
    d = json.loads((LOG_DIR / f"{rid}.json").read_text(encoding="utf-8"))
    print(f"{'='*76}\nrun_id  {d['run_id']}   时间 {d['started_at']}")
    print(f"问题    {d['question']}")
    print(f"结果    {d['steps']} 步、{d['总耗时']}s、{d['tokens']['total']} token、"
          f"工具 {d['tools_used']}")

    print(f"\n★ 决策链（Reason 想什么 → Act 做什么）")
    for s in range(1, d["steps"] + 1):
        llm = next((c for c in d["llm_calls"] if c["step"] == s), {})
        print(f"\n  ── 第 {s} 步（等模型 {llm.get('llm耗时', '?')}s，"
              f"出 {llm.get('completion_tokens', '?')} token）")
        for th in (x for x in d["thoughts"] if x["step"] == s):
            print(f"     [思考] {th['思考'][:200]}")
        for c in (x for x in d["tool_calls"] if x["step"] == s):
            print(f"     [行动] {c['tool']}({c['args']})  耗时 {c['耗时']}s")
            print(f"     [结果] {str(c['result'])[:160]}")

    print(f"\n★ 最终回答\n{d['answer_with_refs'] or d['answer']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        详情(None if sys.argv[1] in ("最后", "last") else sys.argv[1])
    else:
        列表()
