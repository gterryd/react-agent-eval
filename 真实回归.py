r"""第6步-④ 收尾：用真 DeepSeek 验一次，确认改动没弄坏原有功能

★为什么必须有这一步：测重试.py 全是假 client，21 项全绿也只证明了「重试逻辑对」，
  不证明「真实调用还能跑」。我动了 run() 的核心循环，必须真跑一次。
★会花钱（3 次调用，几分钱）。

跑法：python 真实回归.py
"""
import sys

import agent as ag

sys.stdout.reconfigure(encoding="utf-8")
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


if __name__ == "__main__":
    client = ag.new_client()
    print("★用真 DeepSeek 跑 3 道已知答案的题（此前基线里稳定通过的）\n")

    print("① 查配置题（该调 read_config，答 16）")
    t = ag.run(client, "LoRA 的 rank 设的多少", verbose=False)
    check("答案含 16", "16" in t.answer, t.answer[:50])
    check("调了 read_config", "read_config" in t.tools_used, str(t.tools_used))
    check("★没误触降级", t.degraded is False, f"degraded={t.degraded}")
    check("★重试数为0(网络正常)", t.retry_total == 0, f"retry_total={t.retry_total}")

    print("\n② 检索题（该调 search_docs，要有出处）")
    t2 = ag.run(client, "4bit 推理为什么会输出乱码", verbose=False)
    check("调了 search_docs", "search_docs" in t2.tools_used, str(t2.tools_used))
    check("有出处", len(t2.citations) > 0, f"{len(t2.citations)} 条")
    # ★9.27 修正：原断言写的是「来源/参考」，但 append_refs 实际拼的是「出处（可去原文核对）」，
    #   三个词都对不上 → FAIL。★是我的尺子错了，功能一直是好的（第13次同类错，见看板总账）。
    #   改成校验结构：带出处小节 + 文件名 + 比纯答案长。不再靠印象猜措辞。
    check("答案里拼了出处小节", "出处" in t2.answer_with_refs, t2.answer_with_refs[-80:])
    check("出处里有文件名", ".md" in t2.answer_with_refs)
    check("带出处版比纯答案长", len(t2.answer_with_refs) > len(t2.answer),
          f"{len(t2.answer)} → {len(t2.answer_with_refs)}")
    # ★9.27 二次修正：上一版查「出处 not in answer」是 PASS 了，但那是**撞运气**——
    #   模型自己在正文里写的是「（来源：xxx.md）」，措辞不同所以躲过了检查。
    #   ★正确的验法不是猜措辞，而是验结构：append_refs 加的那一段（且只有那一段）不在 answer 里。
    ref_block = t2.answer_with_refs[len(t2.answer):]
    check("★append_refs 是纯追加(没改动原文)",
          t2.answer_with_refs.startswith(t2.answer), f"追加了 {len(ref_block)} 字")
    check("★追加段含分隔标题", "出处（可去原文核对）" in ref_block, ref_block[:30])
    check("★分隔标题不在纯 answer 里", "出处（可去原文核对）" not in t2.answer)
    check("没误触降级", t2.degraded is False)

    print("\n③ 拒答题（知识库外，该说没有而不是编）")
    t3 = ag.run(client, "明天广州天气怎么样", verbose=False)
    check("没编天气", not any(w in t3.answer for w in ["晴", "雨", "°C", "度"]),
          t3.answer[:60])
    check("没误触降级", t3.degraded is False)

    print("\n④ 日志新字段落盘了吗")
    import json
    f = ag.LOG_DIR / f"{t3.run_id}.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    check("详情含 degraded", "degraded" in d, f"degraded={d.get('degraded')}")
    check("详情含 retry_total", "retry_total" in d)
    check("★llm_calls 记了调用次数", "调用次数" in d["llm_calls"][0],
          str(d["llm_calls"][0]))

    print(f"\n{'='*60}")
    print(f"3 题实测：步数 {t.steps}/{t2.steps}/{t3.steps}，"
          f"耗时 {t.elapsed}/{t2.elapsed}/{t3.elapsed}s")
    print("★ 真实回归通过，第6步-④ 完成" if not fails else f"★★ {len(fails)} 项失败：{fails}")
    sys.exit(1 if fails else 0)
