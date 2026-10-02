r"""
第5步 5.2：跑基线，出指标

★★ 每题跑 N 次（默认3次）—— 依据 8.25 实测：同题两次跑行为不同，temperature=0 也压不住。
   跑一次的数字不能当结论，必须报**均值 + 波动**。

指标（照计划书 5.2）：
  任务完成率      答对次数 / 总次数
  ★工具选择准确率  实际调用 ⊇ expected_tools 的比例   ← Agent 特有，RAG 没有
  平均步数/冗余步数 实际步数 - 理论最少步数
  拒答准确率      该拒答的有没有拒
  token 成本 / P95 延迟

跑法见下方 __main__ 提示。结果存 runs.json，供 5.3 归因用。
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from agent import new_client, run
from evalset import ALL_CASES, EASY_TYPES, HARD_TYPES

sys.stdout.reconfigure(encoding="utf-8")

N_RUNS = 3
OUT = Path(__file__).parent / "runs.json"


def judge(case, trace):
    """判定一次运行。★全部由程序算，不许写死结论"""
    missing = [s for s in case.must_have if s not in trace.answer]
    # ★must_any：任一命中即可。判「反问澄清」这类**行为**用它，避免卡措辞（V2/V3 的教训）
    any_ok = (not case.must_any) or any(s in trace.answer for s in case.must_any)
    if not any_ok:
        missing.append(f"未表现出澄清行为(需含任一:{case.must_any[:4]}…)")
    # ★must_not：陷阱题用，抓「附和错误前提」。
    # ★★9.22 加否定语境保护：X4 被误判 3 次的教训 —— 它说「不是 16.6」「16.6 我无法查证」
    #   都被当成附和。所以命中禁止词后，再看前后 12 字内有没有否定/不确定词，有就不算。
    NEGATORS = ("不是", "并非", "而非", "无法", "查不到", "没有", "不能", "不对", "错")
    forbidden = []
    for s in case.must_not:
        pos = trace.answer.find(s)
        while pos >= 0:
            ctx = trace.answer[max(0, pos - 12): pos + len(s) + 12]
            if not any(neg in ctx for neg in NEGATORS):
                forbidden.append(s)      # 附近没否定词 → 才算真附和
                break
            pos = trace.answer.find(s, pos + 1)
    answer_ok = not missing and not forbidden
    if case.type == "reject":
        tool_ok = not trace.tools_used          # 拒答题：一个工具都不该调
    else:
        tool_ok = case.expected_tools.issubset(set(trace.tools_used))
    return {
        "answer_ok": answer_ok,
        "tool_ok": tool_ok,
        "passed": answer_ok and tool_ok,
        "missing": missing,
        "forbidden": forbidden,
        "tools_used": trace.tools_used,
        "steps": trace.steps,
        "elapsed": trace.elapsed,
        "tokens": trace.prompt_tokens + trace.completion_tokens,
        "hit_max_steps": trace.hit_max_steps,
        "answer": trace.answer,
    }


def pct(a, b):
    return f"{a/b*100:.1f}%" if b else "—"


def main():
    client = new_client()
    results = defaultdict(list)
    t_start = time.time()

    for i, case in enumerate(ALL_CASES, 1):
        marks = []
        for k in range(N_RUNS):
            # ★log=False：评测跑 105 次，开日志会刷出 105 个文件。
            #   评测明细本来就存 runs.json，不需要重复落盘
            tr = run(client, case.question, verbose=False, log=False)
            r = judge(case, tr)
            results[case.id].append(r)
            marks.append("o" if r["passed"] else "x")
        got = results[case.id]
        print(f"[{i:>2}/{len(ALL_CASES)}] {case.id:<4}{case.type:<11}"
              f"{''.join(marks)}  {sum(g['passed'] for g in got)}/{N_RUNS} 过  "
              f"步数{[g['steps'] for g in got]}  {case.question[:26]}")

    payload = {"n_runs": N_RUNS,
               "cases": {c.id: {"question": c.question, "type": c.type,
                                "expected_tools": sorted(c.expected_tools),
                                "must_have": c.must_have, "source": c.source,
                                "runs": results[c.id]} for c in ALL_CASES}}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已存 {OUT.name}（供 5.3 失败归因用）  总耗时 {time.time()-t_start:.0f}s")
    return results


def report(results):
    flat = [r for rs in results.values() for r in rs]
    n = len(flat)
    print(f"\n{'='*74}\n★ 总体指标（{len(ALL_CASES)} 题 × {N_RUNS} 次 = {n} 次运行）")
    print(f"  任务完成率        {pct(sum(r['passed'] for r in flat), n)}")
    print(f"  ★工具选择准确率   {pct(sum(r['tool_ok'] for r in flat), n)}")
    print(f"  答案正确率        {pct(sum(r['answer_ok'] for r in flat), n)}")
    print(f"  平均步数          {sum(r['steps'] for r in flat)/n:.2f}")
    print(f"  平均 token        {sum(r['tokens'] for r in flat)/n:.0f}")
    lat = sorted(r["elapsed"] for r in flat)
    print(f"  延迟 中位/P95     {lat[n//2]:.2f}s / {lat[int(n*0.95)-1]:.2f}s")
    print(f"  撞上限次数        {sum(r['hit_max_steps'] for r in flat)}")

    # ★分批看：首轮20题(简单) vs 9.21加难那批。混在一起算平均会把问题平均掉
    print(f"\n★★ 简单批 vs 加难批（★这才是能不能测出问题的关键对照）")
    for label, types in (("简单批(首轮20题)", EASY_TYPES), ("★加难批(9.21)", HARD_TYPES)):
        rs = [r for c in ALL_CASES if c.type in types for r in results[c.id]]
        nq = len([c for c in ALL_CASES if c.type in types])
        print(f"  {label:<18}{nq:>2}题  完成率 {pct(sum(r['passed'] for r in rs), len(rs)):>7}"
              f"   工具准确 {pct(sum(r['tool_ok'] for r in rs), len(rs)):>7}"
              f"   平均步数 {sum(r['steps'] for r in rs)/len(rs):.2f}")

    print(f"\n★ 分类型（看瓶颈在哪类题）")
    by = defaultdict(list)
    for c in ALL_CASES:
        by[c.type] += results[c.id]
    for t, rs in by.items():
        flag = " ←加难" if t in HARD_TYPES else ""
        print(f"  {t:<14}完成率 {pct(sum(r['passed'] for r in rs), len(rs)):>7}   "
              f"工具准确 {pct(sum(r['tool_ok'] for r in rs), len(rs)):>7}   "
              f"平均步数 {sum(r['steps'] for r in rs)/len(rs):.2f}{flag}")

    print(f"\n★★ 稳定性（同题 {N_RUNS} 次结果不一致 = 不稳定，这是 8.25 发现的问题）")
    unstable = [c.id for c in ALL_CASES
                if len(set(r["passed"] for r in results[c.id])) > 1]
    print(f"  不稳定题数 {len(unstable)}/{len(ALL_CASES)}"
          + (f"：{unstable}" if unstable else ""))

    fails = [(c.id, c.type, r) for c in ALL_CASES for r in results[c.id] if not r["passed"]]
    if fails:
        print(f"\n★ 失败清单（{len(fails)} 次，5.3 要逐条归因）")
        for cid, typ, r in fails:
            why = []
            if not r["tool_ok"]:
                why.append(f"工具={r['tools_used']}")
            if r["missing"]:
                why.append(f"答案缺{r['missing']}")
            if r.get("forbidden"):
                why.append(f"★出现禁止内容{r['forbidden']}")
            print(f"  {cid:<4}{typ:<11}{'; '.join(why)}")


if __name__ == "__main__":
    report(main())
