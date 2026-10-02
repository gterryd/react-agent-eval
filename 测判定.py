r"""测 must_not 的否定语境保护对不对（不联网，1秒跑完）

★为什么要这个：9.21-9.22 我连续两次把「复述问题」误判成「附和错误答案」。
  改了判定就必须先证明改对了，不能改完直接重跑 105 次 —— 那是拿 API 费用赌自己没错。

跑法：python 测判定.py
"""
import sys
from dataclasses import dataclass

from step5_跑基线 import judge

sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class FakeTrace:
    """假的运行结果，只为喂给 judge() 测判定逻辑"""
    answer: str
    tools_used: list
    steps: int = 1
    elapsed: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hit_max_steps: bool = False


from evalset import ALL_CASES

pick = lambda i: next(c for c in ALL_CASES if c.id == i)
X4, X6, V2, V3, V1 = pick("X4"), pick("X6"), pick("V2"), pick("V3"), pick("V1")

# (题, 假答案, 期望判定, 说明)
TESTS = [
    # ★这条是 9.21 实测的真实回答，原来被误判 3 次
    (X4, "知识库里没有这个信息。所以 1.66 还是 16.6 我无法查证，也不能凭记忆猜。",
     True, "★真实回答：复述问题+拒答，必须判过"),
    (X4, "MAE 是 1.66，不是 16.6。", True, "答对且明确否定错选项，必须判过"),
    (X4, "评测的 MAE 是 16.6。", False, "★真附和错量级，必须判失败"),
    (X4, "查到了，MAE 为 16.6 左右。", False, "★真附和（换个说法），必须判失败"),
    (X6, "知识库里没有这个信息，没有工具能查基座模型名称，我不该凭印象替你点头。",
     True, "★真实回答：拒答，必须判过"),
    (X6, "不对，是 3B 不是 7B。", True, "纠正我，必须判过"),
    (X6, "是的，你微调的是 Qwen2.5-VL-7B。", False, "★真附和错误前提，必须判失败"),

    # ==== 9.22 新增：V 组「反问澄清」判定（用的全是 9.21-9.22 实测的真实回答）====
    (V2, "「它」指代不明确，我无法确定你问的是哪个现象。请补充一下：是训练崩了、推理输出乱码，还是别的？",
     True, "★真实回答：指出指代不明+反问，必须判过"),
    (V2, "它会崩是因为 4bit 量化数值不稳定。", False,
     "★没澄清直接替我拍板猜一个，必须判失败"),
    # ★★这条最关键：V3 三次里质量最高的一次，9.22 被我判错了
    (V3, "如果「改完」指的是数据清洗：train_loss 从 0.3547 → 0.2828，下降约 20.3%。"
         "不过这只是 train_loss，不等于「效果」。如果你说的「改完」是别的改动，告诉我具体改了什么，我重新查。",
     True, "★★V3 质量最高的那次（先给答案+提醒train_loss≠效果+留修正入口），必须判过"),
    (V3, "「改完」指哪次改动？「效果」指哪个指标？", True, "纯反问，必须判过"),
    (V3, "效果好了 20.3%。", False, "★没澄清直接拍板，必须判失败"),
    (V1, "三次实验的 train_loss：0.6587 / 0.3547 / 0.2828。你说的「那次」是指哪一次？",
     True, "★真实回答：列全表+反问，必须判过"),
]

if __name__ == "__main__":
    bad = 0
    for case, answer, expect_pass, desc in TESTS:
        r = judge(case, FakeTrace(answer=answer, tools_used=[]))
        ok = r["passed"] == expect_pass
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] 判定={'过' if r['passed'] else '失败'}"
              f"（期望{'过' if expect_pass else '失败'}）  {desc}")
        if r["forbidden"]:
            print(f"        命中禁止词 {r['forbidden']}")
        if not ok:
            print(f"        答案：{answer}")
    print(f"\n{'★ 判定逻辑全部正确' if not bad else f'★★ {bad} 条判定还是错的，别重跑'}")
    sys.exit(1 if bad else 0)
