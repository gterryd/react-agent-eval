r"""
【一次性小实验】证明一件事：模型调不调工具，只看「说明书(description)」怎么写

同一个问题「rank 是多少」，同一个工具函数，只改说明书，跑三种写法看差别。

跑法（cmd 或 PowerShell 都行）：
  cd <项目目录>
  python 实验_改说明书.py
"""
import sys

from step2_tool_calls import CONFIG, load_env, read_config
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")


def make_tools(description):
    """只有 description 不一样，函数本身完全没变"""
    return [{
        "type": "function",
        "function": {
            "name": "read_config",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }]


VERSIONS = [
    ("原版：说明书里明确写了可查 rank",
     "查询本次大模型微调训练的配置参数值。可查：batch_size、learning_rate、rank、alpha、epoch。"),
    ("★删掉 rank：说明书只剩别的参数",
     "查询本次大模型微调训练的配置参数值。可查：batch_size、learning_rate、epoch。"),
    ("★说明书写成一句废话",
     "一个工具。"),
]

if __name__ == "__main__":
    load_env()
    import os
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                    base_url=os.environ["DEEPSEEK_BASE_URL"])

    question = "训练用的 rank 是多少"
    for title, desc in VERSIONS:
        msg = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": question}],
            tools=make_tools(desc),
            temperature=0,
        ).choices[0].message

        print("=" * 60)
        print(f"【{title}】")
        print(f"说明书内容：{desc}")
        if msg.tool_calls:
            import json
            tc = msg.tool_calls[0]
            key = json.loads(tc.function.arguments).get("key")
            real = read_config(key)  # ★真的去查一次，别猜
            print(f"→ 调了工具，参数 = {tc.function.arguments}")
            print(f"→ ★真实查询结果：read_config({key!r}) = {real}")
        else:
            print(f"→ 结果：★没调工具，它直接自己回答了：")
            print(f"   {msg.content[:150]}")
        print()
