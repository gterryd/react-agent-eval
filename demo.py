r"""第9步-① Gradio 演示界面

★设计原则：**不只给答案，要能看见它是怎么想的。**
  这是 Agent 项目，决策链才是主角 —— 给个聊天框只能看到答案，
  那和普通问答机器人没区别，看不出 Agent 在哪。
  所以界面分 4 块：答案 / 决策链（第几步想什么、调了什么）/ 出处 / 本次指标。

★两版可切换（手写 while 版 vs LangGraph 图版）—— 第8.5步的对照直接做进界面，
  演示时现场切，比口头说「我两版都实现了」有说服力。

⚠️ 安全：`share=False`，只绑本机。无鉴权，不要开公网分享链接。

启动（cmd，先 cd 到项目目录）：
  python demo.py
然后浏览器打开 http://127.0.0.1:7860
"""
import os

# ⚠️★★ 这三行必须在所有 import 之前，位置不能动（10.2 踩坑实录）：
#   症状：启动卡在「连 huggingface.co 超时，Retrying in 1s [Retry 1/5]」白等。
#   原因：gradio 会连带导入 huggingface_hub，而 huggingface_hub **在被导入的那一刻**
#        就把 ENDPOINT 读成模块常量了。我原本只在 vec.py 里设镜像，
#        但 demo.py 先 import gradio → 等 vec.py 再设已经晚了。
#   ★实测证据：导入gradio后 ENDPOINT=huggingface.co；事后设环境变量+reload 才变成镜像。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import sys
import time

import gradio as gr

import agent as ag
import graph_agent as gg
import tools as tl

sys.stdout.reconfigure(encoding="utf-8")

_client = None


def get_client():
    global _client
    if _client is None:
        _client = ag.new_client()
    return _client


def 画决策链(t) -> str:
    """★把 Trace 翻译成人能看懂的一步一步。这块是整个界面的重点。"""
    if not t.thoughts and not t.calls:
        return "（这次没有中间步骤）"
    out = []
    for step in range(1, t.steps + 1):
        想 = next((x["思考"] for x in t.thoughts if x["step"] == step), "")
        做 = [c for c in t.calls if c["step"] == step]
        out.append(f"### 第 {step} 步")
        if 想:
            out.append(f"**它在想**：{想}")
        for c in 做:
            out.append(f"**它决定调工具**：`{c['tool']}`　参数 `{c['args']}`"
                       f"　（耗时 {c['耗时']}s）")
            out.append(f"> 工具返回：{str(c['result'])[:300]}"
                       f"{'…（截断）' if len(str(c['result'])) > 300 else ''}")
        if not 做 and 想:
            out.append("**没有调工具** → 它认为可以直接回答了（★这就是 ReAct 的退出条件）")
        out.append("")
    return "\n\n".join(out)


def 画出处(t) -> str:
    if not t.citations:
        return "（这次没用检索工具，所以没有出处）"
    rows = ["| # | 来自哪份笔记 | 哪一节 | 相似度 |", "|---|---|---|---|"]
    for i, c in enumerate(sorted(t.citations, key=lambda x: -x["score"]), 1):
        rows.append(f"| {i} | {c['source']} | {c['heading']} | {c['score']} |")
    return "\n".join(rows)


def 画指标(t, 版本, 墙上时间) -> str:
    冷 = [c for c in t.calls if c["耗时"] >= 3.0]      # ★9.24 教训：冷启动要单独标
    行 = [f"- **编排版本**：{版本}",
         f"- **走了几步**：{t.steps}（上限 {ag.MAX_STEPS}，撞上限：{t.hit_max_steps}）",
         f"- **用了哪些工具**：{t.tools_used or '一个都没用'}",
         f"- **总耗时**：{墙上时间:.2f}s",
         f"- **token**：输入 {t.prompt_tokens} + 输出 {t.completion_tokens} "
         f"= {t.prompt_tokens + t.completion_tokens}",
         f"- **API 重试次数**：{t.retry_total}　**是否降级**：{t.degraded}",
         f"- **run_id**：`{t.run_id}`（日志在 logs/ 下，可回查）"]
    if 冷:
        行.append(f"- ⚠️ **含冷启动 {冷[0]['耗时']}s**（首次加载 BGE 向量模型，"
                 f"不是检索慢 —— ★这个坑我 9.24 栽过一次）")
    return "\n".join(行)


def 问(question, 版本):
    if not question or not question.strip():
        return "请先输入问题", "", "", ""
    runner = gg.run if 版本.startswith("LangGraph") else ag.run
    t0 = time.time()
    try:
        t = runner(get_client(), question.strip(), verbose=False)
    except Exception as e:      # ★界面不能崩，出错也要显示出来
        return f"**出错了**：{type(e).__name__}: {e}", "", "", ""
    墙 = time.time() - t0
    return t.answer or "（没有返回答案）", 画决策链(t), 画出处(t), 画指标(t, 版本, 墙)


示例 = [
    ["LoRA 的 rank 设的多少", "手写版（while 循环）"],
    ["4bit 推理为什么会输出乱码", "手写版（while 循环）"],
    ["三次实验里哪次 loss 最低，训练了多少小时", "LangGraph 版（状态图）"],
    ["干净数据那次训练，平均每步要多少秒", "手写版（while 循环）"],
    ["明天广州天气怎么样", "手写版（while 循环）"],
]

说明 = """# VLM 项目问答 Agent

问我关于「我自己那个 VLM 微调项目」的事。它会**自己决定**要不要查、查什么、查几次。

- **5 个工具**：笔记检索(BGE向量) / 查配置 / 查实验记录 / 算数 / 列文件
- **两种编排**：手写 while 循环 ｜ LangGraph 状态图（★右边可切，行为已验证一致）
- ★**右侧「它是怎么想的」是重点** —— 答案谁都能给，Agent 的价值在决策过程
- 问知识库外的事（比如天气），它会说没有，★不会编
"""


def build():
    with gr.Blocks(title="VLM 项目问答 Agent", theme=gr.themes.Soft()) as ui:
        gr.Markdown(说明)
        with gr.Row():
            with gr.Column(scale=3):
                q = gr.Textbox(label="你的问题", lines=2,
                               placeholder="例：LoRA 的 rank 设的多少")
            with gr.Column(scale=2):
                版本 = gr.Radio(["手写版（while 循环）", "LangGraph 版（状态图）"],
                              value="手写版（while 循环）", label="编排方式")
        发问 = gr.Button("问它", variant="primary")
        with gr.Row():
            with gr.Column(scale=1):
                答案 = gr.Markdown(label="答案")
            with gr.Column(scale=1):
                with gr.Tab("它是怎么想的（决策链）"):
                    链 = gr.Markdown()
                with gr.Tab("出处（可去原文核对）"):
                    源 = gr.Markdown()
                with gr.Tab("本次指标"):
                    标 = gr.Markdown()
        gr.Examples(examples=示例, inputs=[q, 版本], label="点一个试试")
        for ev in (发问.click, q.submit):
            ev(问, inputs=[q, 版本], outputs=[答案, 链, 源, 标])
    return ui


if __name__ == "__main__":
    print("（预热：加载 BGE 向量模型，约 15s，之后提问就快了）")
    t0 = time.time()
    try:
        tl.search_docs("预热")
        print(f"（预热完成 {time.time()-t0:.1f}s）")
    except Exception as e:
        print(f"（预热失败，首次提问会慢：{e}）")
    # ⚠️ share=False：只绑本机。★无鉴权，不要开公网分享链接。
    build().launch(server_name="127.0.0.1", server_port=7860,
                   share=False, inbrowser=True)
