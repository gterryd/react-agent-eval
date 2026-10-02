r"""
第3步验收：用第1步定的 5 道种子题跑 ReAct 循环

过关标准（计划书第3步）：③④ 能走两步答对、⑤ 能拒答

★ 判定规则全部由程序执行，不由我口头说。任一条 FAIL → 退出码 1。
跑法：
  cd <项目目录>
  python step3_验收.py
"""
import sys

from agent import new_client, run

sys.stdout.reconfigure(encoding="utf-8")

# (编号, 问题, 答案里必须出现的字符串, 期望用到的工具集合, 是否应该拒答)
CASES = [
    # ★9.2 修正判定：原先要求含 "dtype"，但那是我第3步编假文档时的说法。
    # 真实笔记（技术复盘记录.md）写的是「4bit 推理数值不稳定 / bitsandbytes 精度不够」。
    # ★教训：判定标准必须以真实资料为准，不能拿自己编的答案去判对错。
    ("①单-检索", "4bit 推理为什么输出乱码", ["4bit"], {"search_docs"}, False),
    ("②单-配置", "LoRA 的 rank 设的多少", ["16"], {"read_config"}, False),
    ("③★多步", "train_loss 最低的那次实验，训练了多久、走了多少步",
     ["6", "56", "450"], {"read_experiment"}, False),
    ("④★需计算", "139 个比较项里，严格准确率 30.22% 对应多少项？取整数",
     ["42"], {"calculate"}, False),
    ("⑤★应拒答", "今天广州天气怎么样", ["没有"], set(), True),
]

if __name__ == "__main__":
    client = new_client()
    fails, rows = [], []

    for tag, q, must_have, expect_tools, should_reject in CASES:
        print(f"\n{'='*66}\n{tag}  问：{q}")
        t = run(client, q)

        # 三条判定，全部由程序算
        missing = [s for s in must_have if s not in t.answer]
        tool_ok = (not t.tools_used) if should_reject else expect_tools.issubset(set(t.tools_used))
        ok = not missing and tool_ok

        print(f"  回答：{t.answer[:160]}")
        print(f"  → 工具序列={t.tools_used}  步数={t.steps}  耗时={t.elapsed}s  "
              f"token={t.prompt_tokens}+{t.completion_tokens}")
        if not tool_ok:
            print(f"  ✗ 工具不符：期望包含 {expect_tools or '（不调工具）'}")
        if missing:
            print(f"  ✗ 答案缺少：{missing}")
        print(f"  【{'PASS' if ok else 'FAIL'}】")

        if not ok:
            fails.append(tag)
        rows.append((tag, t, ok))

    print(f"\n{'='*66}\n汇总（这张表就是第5步指标的雏形）")
    print(f"{'题':<12}{'结果':<7}{'步数':<6}{'工具数':<7}{'耗时s':<8}{'token'}")
    for tag, t, ok in rows:
        print(f"{tag:<12}{'PASS' if ok else 'FAIL':<7}{t.steps:<6}"
              f"{len(t.tools_used):<7}{t.elapsed:<8}{t.prompt_tokens + t.completion_tokens}")

    total_tok = sum(t.prompt_tokens + t.completion_tokens for _, t, _ in rows)
    print(f"\n任务完成率 = {len(rows)-len(fails)}/{len(rows)} = "
          f"{(len(rows)-len(fails))/len(rows)*100:.0f}%    总 token = {total_tok}")

    if fails:
        print(f"\n★★ 有 {len(fails)} 条 FAIL：{fails}  ← 别信这一步过关了")
        sys.exit(1)
    print("\n★ 5 道种子题全过，第3步过关。以上数字全部程序当场算出")
