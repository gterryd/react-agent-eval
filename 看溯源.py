r"""第6步-① 引用溯源：看效果

★这一步解决的问题：「你怎么知道它没瞎说？」
  每个回答末尾附上「出自哪个文件、哪一节」，可以去原文核对。

跑法（cmd，先 cd 到项目目录）：
  python 看溯源.py
"""
import sys

from agent import new_client, run

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "4bit 推理为什么会输出乱码",          # 检索题 → 应该有出处
    "LoRA 的 rank 设的多少",              # 查配置 → 没走检索，应该没出处
    "哪些类别数得准、哪些数不准",          # 检索题 → 应该有出处
]

if __name__ == "__main__":
    client = new_client()
    for q in QUESTIONS:
        print("=" * 76)
        print(f"问：{q}\n")
        t = run(client, q, verbose=False)
        print(t.answer_with_refs or t.answer)
        print(f"\n（用了工具 {t.tools_used}；收集到 {len(t.citations)} 条出处）\n")
