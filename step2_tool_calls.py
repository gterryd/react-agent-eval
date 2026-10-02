r"""
第2步：第一个工具 + tool calls 跑通
—— 目标：让 DeepSeek 自己决定「要不要调工具、调哪个、传什么参数」

跑法：
  python step2_tool_calls.py

过关标准（计划书第2步）：
  ① 问「rank 是多少」→ 它自己调 read_config("rank") → 答出 16
  ② 问「今天天气」  → 它不调这个工具
"""
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，打不出 ✅ 这类字符


def load_env(path=Path(__file__).parent / ".env"):
    """极简 .env 读取，省一个依赖。key 不写死在代码里，防止推 GitHub 泄露"""
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ---------------- 真正干活的函数 ----------------
# 第2步先用写死的字典，第4步才换成真读文件。数字出自 adapter_config.json / 训练配置
CONFIG = {"batch_size": 2, "learning_rate": 1e-4, "rank": 16, "alpha": 32, "epoch": 3}


def read_config(key: str) -> str:
    return str(CONFIG.get(key, f"没有 {key} 这个参数"))


# ---------------- Tool Schema：用 JSON 告诉模型「我有什么工具」 ----------------
# ★description 是模型判断该不该调的唯一依据，写清楚它管什么、不管什么
TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_config",
        "description": "查询本次大模型微调训练的配置参数值。可查：batch_size、learning_rate、rank、alpha、epoch。只管训练配置，不管天气、新闻等无关问题。",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "参数名，如 rank"}},
            "required": ["key"],
        },
    },
}]

DISPATCH = {"read_config": read_config}


def ask(client, question: str, max_steps: int = 5):
    """返回 (最终回答, 实际调用的工具名列表)"""
    messages = [{"role": "user", "content": question}]
    used = []
    for _ in range(max_steps):  # ★必须设上限，否则模型可能反复调同一个工具
        msg = client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=TOOLS, temperature=0
        ).choices[0].message
        messages.append(msg)
        if not msg.tool_calls:  # 没有工具调用 = 它要直接回答了
            return msg.content, used
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = DISPATCH[tc.function.name](**args)
            used.append(f"{tc.function.name}({args})")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "达到步数上限", used


if __name__ == "__main__":
    load_env()
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"])

    cases = [
        ("训练用的 rank 是多少", True),        # 该调
        ("今天天气怎么样", False),             # 该拒答
        ("batch size 和 epoch 分别是多少", True),  # 该调两次（测并行/连续调用）
        ("optimizer 用的是什么", True),        # 该调但字典里没有 → 测「查不到」时会不会编
        ("1+1 等于几", False),                 # 无关问题，别滥用工具
    ]
    for q, should_call in cases:
        answer, used = ask(client, q)
        ok = "[通过]" if bool(used) == should_call else "[不通过]"
        print(f"\n{ok} 问：{q}")
        print(f"   调用工具：{used if used else '（未调用）'}   期望调用：{should_call}")
        print(f"   回答：{answer}")
