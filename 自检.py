r"""
★★ 自检脚本：验证「文档里声称的结论」是不是真的

存在的理由（8.25）：
  8.24 我在 实验_改说明书.py 里写死了 print(CONFIG.get('rank'))，
  导致参数填错(training.rank)时屏幕照样显示「能查到 16」——**假造了结果**。

铁律（写进代码，不靠承诺）：
  1. 屏幕上每个结论都必须是**程序当场算出来的**，不许预先写死
  2. 每条用例都有 expected，程序自己判 PASS/FAIL，**不由我口头说**
  3. 有任何一条 FAIL → 退出码非 0，**响亮地失败**，不许静悄悄过去

跑法（cmd 或 PowerShell 都行）：
  cd <项目目录>
  python 自检.py
"""
import json
import os
import sys

from openai import OpenAI

from step2_tool_calls import CONFIG, TOOLS, ask, load_env, read_config

sys.stdout.reconfigure(encoding="utf-8")

FAILS = []


def check(name, actual, expected):
    """唯一的判定入口。actual 必须是真实跑出来的值，不许传字面量"""
    ok = actual == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         实际={actual!r}")
    if not ok:
        print(f"         期望={expected!r}   ← ★不一致")
        FAILS.append(name)
    return ok


def part1_纯函数():
    """不联网，验证工具函数本身。这部分是确定性的，每次结果必须一样"""
    print("\n【第1部分】工具函数本身（不联网，结果必须每次相同）")
    check("read_config('rank')", read_config("rank"), "16")
    check("read_config('alpha')", read_config("alpha"), "32")
    check("read_config('epoch')", read_config("epoch"), "3")
    # ★8.24 骗人的那条：参数写错时必须查不到，不许返回 16
    check("read_config('training.rank') 必须查不到",
          read_config("training.rank"), "没有 training.rank 这个参数")


def part2_模型行为(client):
    """联网。★模型输出有随机性，所以只断言「有没有调工具」这种稳定的事实，
    不断言回答的具体措辞（那个每次都不一样，断言了会假失败）"""
    print("\n【第2部分】模型会不会调工具（联网，跑真实 API）")
    for q, expect_call in [("训练用的 rank 是多少", True),
                           ("今天天气怎么样", False),
                           ("1+1 等于几", False)]:
        answer, used = ask(client, q)
        check(f"问「{q}」是否调工具", bool(used), expect_call)
        if used:
            print(f"         调用记录={used}")
        print(f"         回答前60字={answer[:60]!r}")


def part3_答案里必须真有那个数(client):
    """★端到端：问 rank，最终回答的文字里必须真的出现 16。
    这条能抓住「工具查对了但汇总时说错」——第5步归因表的第⑤类"""
    print("\n【第3部分】最终回答里是否真的含正确数字（端到端）")
    answer, used = ask(client, "训练用的 rank 是多少")
    check("回答文字里含 '16'", "16" in answer, True)
    check("回答里不含错误值 '8'", "8" in answer.replace("16", ""), False)
    print(f"         完整回答={answer!r}")


if __name__ == "__main__":
    load_env()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"])
    part1_纯函数()
    part2_模型行为(client)
    part3_答案里必须真有那个数(client)

    print("\n" + "=" * 60)
    if FAILS:
        print(f"★★ 有 {len(FAILS)} 条 FAIL，别信这一步的结论：")
        for f in FAILS:
            print(f"   - {f}")
        sys.exit(1)  # ★退出码非0，响亮地失败
    print("★ 全部 PASS。以上每个数字都是程序当场算的，没有一个是写死的")
