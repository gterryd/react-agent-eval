r"""第8.5步验收：手写版 vs LangGraph 版，同题同问跑对比

★这个脚本才是第8.5步真正的产出。
  光说「我用了 LangGraph」谁都会说；有一份对照数据，才能回答
  「框架帮你省了什么、藏了什么」「不用框架你自己写得出来吗」。

★两版必须行为一致：同一道题，答案要点、工具选择、步数应当对得上。
  （★耗时和 token 不会完全相同 —— 模型输出本身有波动，8.25 实测过同码两跑不同。）

跑法：python 对照_手写vs图.py
★会真调 API：4题×2版=8次调用，几分钱。
"""
import sys

import agent as ag
import graph_agent as gg

sys.stdout.reconfigure(encoding="utf-8")
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


# 4 道题覆盖 4 种路径：单工具 / 多步依赖 / 需计算 / 应拒答
CASES = [
    {"q": "LoRA 的 rank 设的多少", "must": ["16"],
     "tools": {"read_config"}, "型": "单工具"},
    # ★9.28 查过真实数据才写断言（不再凭印象，见看板总账9.27教训）：
    #   最低 loss = 干净数据全量 0.2828，时长字段是文字「6小时56分」(=6.94h，训练秒数24987)。
    #   ★所以这题**不强求调 calculate** —— 时长本来就能直接读出来，换算是可选的。
    #   必须答对的是「哪次」和「几小时」，用 任一 放行 6小时56分 / 6.94 / 6.9 三种写法。
    {"q": "三次实验里哪次 loss 最低，训练了多少小时",
     "must": ["6小时56", "6.94", "6.9"], "任一": True,
     "tools": {"read_experiment"}, "型": "多步", "必含": "干净"},
    {"q": "4bit 推理为什么会输出乱码", "must": [],
     "tools": {"search_docs"}, "型": "检索"},
    {"q": "明天广州天气怎么样", "must": [],
     "tools": set(), "型": "应拒答", "禁": ["晴", "雨", "°C"]},
]


def 跑一版(name, runner, client):
    print(f"\n{'='*60}\n★ {name}")
    rows = []
    for c in CASES:
        t = runner(client, c["q"], verbose=False, log=False)
        rows.append(t)
        print(f"\n  [{c['型']}] {c['q']}")
        print(f"    步数{t.steps} 工具{t.tools_used} {t.elapsed}s "
              f"token{t.prompt_tokens + t.completion_tokens} 降级{t.degraded}")
        print(f"    答：{(t.answer or '')[:70]}")
    return rows


def 验正确性(label, rows):
    """★先各自验对不对，再比两版是否一致。顺序不能颠倒 ——
    两版都错但错得一样，也会「一致」，那种一致没有意义。"""
    print(f"\n★ {label} 正确性")
    for c, t in zip(CASES, rows):
        a = t.answer or ""
        if c["must"]:
            ok = any(m in a for m in c["must"]) if c.get("任一") \
                else all(m in a for m in c["must"])
            check(f"{c['型']}·答案含{c['must']}", ok, a[:40])
        if c.get("必含"):        # ★还得答对「是哪次实验」，不能只答个数字
            check(f"{c['型']}·答出是哪次({c['必含']})", c["必含"] in a, a[:40])
        if c.get("禁"):
            check(f"{c['型']}·没编内容", not any(w in a for w in c["禁"]), a[:40])
        if c["tools"]:
            check(f"{c['型']}·调了{c['tools']}", c["tools"].issubset(set(t.tools_used)),
                  str(t.tools_used))
        else:
            check(f"{c['型']}·一个工具都没调", not t.tools_used, str(t.tools_used))
        check(f"{c['型']}·没误触降级", t.degraded is False)


if __name__ == "__main__":
    client = ag.new_client()
    print("★图结构：")
    print(gg.GRAPH.get_graph().draw_ascii())

    手 = 跑一版("手写版 agent.py（while 循环）", ag.run, client)
    图 = 跑一版("LangGraph 版 graph_agent.py（状态图）", gg.run, client)

    验正确性("手写版", 手)
    验正确性("图版本", 图)

    print("\n★ 两版行为一致性（★工具选择必须一致；耗时/token 允许波动）")
    for c, a, b in zip(CASES, 手, 图):
        check(f"{c['型']}·工具选择一致",
              set(a.tools_used) == set(b.tools_used),
              f"手写{a.tools_used} vs 图{b.tools_used}")
        check(f"{c['型']}·步数一致", a.steps == b.steps, f"{a.steps} vs {b.steps}")

    print(f"\n{'='*60}\n★ 对照表（贴进看板和 README）")
    print(f"{'题型':<10}{'手写步数':>8}{'图步数':>8}{'手写s':>9}{'图s':>9}"
          f"{'手写token':>11}{'图token':>10}")
    for c, a, b in zip(CASES, 手, 图):
        print(f"{c['型']:<10}{a.steps:>8}{b.steps:>8}{a.elapsed:>9}{b.elapsed:>9}"
              f"{a.prompt_tokens+a.completion_tokens:>11}"
              f"{b.prompt_tokens+b.completion_tokens:>10}")
    print(f"\n{'='*60}")
    print("★ 两版行为一致，第8.5步通过" if not fails else f"★★ {len(fails)} 项失败：{fails}")
    sys.exit(1 if fails else 0)
