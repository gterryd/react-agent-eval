r"""第8.5步：用 LangGraph 重写编排（★与手写版 agent.py 并存，一行都没删）

★为什么重写：
  1. 目标岗位的 JD 里点名要 LangGraph，这是最直接的原因，不回避
  2. 图结构比 while 好扩展（第7步多轮、第8步双Agent 加节点就行）
  3. ★最值钱的：我有「手写版 vs 框架版」的对照 —— 先手写跑通拿到 93.3% 基线，
     再用框架重写。所以我能回答「框架帮我省了什么、藏了什么」，
     而不是只会说「我用了 LangGraph」。

★逻辑和 agent.py 完全一样，只是换了表达方式：
  手写版： for step in range(6):  问模型 → 要调工具吗 → 调 → 塞回去 → 再问
  图版本： [think 节点] --条件边--> [act 节点] --回边--> [think 节点] ... --> END

跑法：python graph_agent.py
"""
import json
import sys
import time
import uuid
from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

import agent as ag          # ★直接复用手写版的 Trace/call_llm/save_log/append_refs
from tools import DISPATCH, TOOLS

sys.stdout.reconfigure(encoding="utf-8")


def _append(old: list, new) -> list:
    """归约函数：节点返回的消息**追加**到已有列表，而不是覆盖它。
    ★保持 openai 原生格式不转换（踩坑记录见 AgentState.messages 注释）。"""
    if new is None:
        return old
    return list(old) + (list(new) if isinstance(new, list) else [new])


class AgentState(TypedDict):
    """★图的「状态」：节点之间传的就是这个东西。

    ★手写版里这些是散落的局部变量（messages 列表 + Trace 对象），
      图版本必须先把状态**显式声明**出来 —— 这是框架强加的约束，
      一开始觉得啰嗦，但第8步做双 Agent 时，状态清晰就是救命的。
    """
    # ★「归约函数」：节点返回新消息时自动追加而不是覆盖。这是框架帮我省事的地方 ——
    #   手写版我得自己记得 messages.append()。
    # ⚠️★9.28 实测踩坑：本来用官方的 add_messages，结果 DeepSeek 报
    #   「messages[0]: missing field `role`」。查明原因：add_messages 会把我的 dict
    #   **悄悄转成 LangChain 的 SystemMessage/HumanMessage 对象，这些对象没有 role 属性**，
    #   而 call_llm 是直接喂给 openai SDK 的，它只认 {"role":..,"content":..}。
    # ★选择：不用 add_messages，自己写 6 行 append。理由是本项目全链路
    #   （call_llm / 工具消息 / api.py）都是 openai 原生格式，为一个归约器把整条链
    #   改成 LangChain 生态不值得 —— ★**框架该用哪部分是我选的，不是它说什么我都吞。**
    messages: Annotated[list, _append]
    trace: ag.Trace         # 复用手写版的 Trace，日志/溯源/重试统计全部照旧
    max_steps: int
    verbose: bool
    _client: object         # OpenAI client，随状态传给节点（节点是纯函数，不抓全局）


def think(state: AgentState) -> dict:
    """节点①：问模型「你还要调工具吗」。★对应 ReAct 的 Reason。

    ★节点只返回「要改哪些状态」，不自己 mutate —— 框架负责合并。
    """
    t, step = state["trace"], state["trace"].steps + 1
    c0 = time.time()
    try:
        # ★重试与降级直接复用第6步-④ 的 call_llm，没有重复实现
        resp, attempts, waited = ag.call_llm(state["_client"], state["messages"])
    except ag.LLMUnavailable as e:
        t.degraded, t.degrade_reason = True, str(e)
        t.answer = f"抱歉，{e}。请稍后再试。"
        t.answer_with_refs = t.answer
        t.llm_calls.append({"step": step, "llm耗时": round(time.time() - c0, 3),
                            "失败": str(e)})
        return {}               # 不加消息 → 条件边会看到 degraded 直接收尾

    msg = resp.choices[0].message
    t.steps = step
    t.prompt_tokens += resp.usage.prompt_tokens
    t.completion_tokens += resp.usage.completion_tokens
    t.retry_total += attempts - 1
    t.llm_calls.append({"step": step, "llm耗时": round(time.time() - c0, 3),
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "finish_reason": resp.choices[0].finish_reason,
                        "调用次数": attempts, "重试等待": waited})
    if msg.content:
        t.thoughts.append({"step": step, "思考": msg.content})
    return {"messages": [msg]}


def act(state: AgentState) -> dict:
    """节点②：执行模型点名的工具。★对应 ReAct 的 Act。"""
    t = state["trace"]
    last = state["messages"][-1]
    out = []
    for tc in last.tool_calls:
        args = json.loads(tc.function.arguments or "{}")
        r0 = time.time()
        try:
            result = DISPATCH[tc.function.name](**args)
        except Exception as e:      # ★工具挂了不能让整个 Agent 崩
            result = f"工具执行失败：{e}"
        cost = time.time() - r0
        if tc.function.name == "search_docs":   # ★第6步-① 收出处
            _collect_citations(t)
        t.tools_used.append(tc.function.name)
        t.calls.append({"step": t.steps, "tool": tc.function.name,
                        "args": args, "result": result, "耗时": round(cost, 3)})
        if state["verbose"]:
            print(f"  第{t.steps}步 → {tc.function.name}({args})")
            print(f"         得到：{str(result)[:100]}")
        out.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
    return {"messages": out}


def _collect_citations(t):
    """出处去重，保留最高分。★和手写版同一套规则，抽出来避免两份不一致"""
    import tools as _t
    for c in _t.LAST_CITATIONS:
        same = next((x for x in t.citations if x["source"] == c["source"]
                     and x["heading"] == c["heading"]), None)
        if same is None:
            t.citations.append(dict(c))
        elif c["score"] > same["score"]:
            same["score"] = c["score"]


def should_continue(state: AgentState) -> str:
    """★条件边：整个 Agent 的「自主性」就在这个函数里。

    ★手写版里这是 `if not msg.tool_calls: break` 那一行 ——
      藏在循环中间，看不出它是个决策点。
    ★图版本被迫把它单拎出来命名，路径变成可画、可讲的东西。
      这是重写最大的收获：**决策点显式化**。
    """
    t = state["trace"]
    if t.degraded:                          # 模型服务挂了 → 收尾
        return "end"
    if t.steps >= state["max_steps"]:       # ★撞上限 → 收尾（防死循环烧钱）
        t.hit_max_steps = True
        t.answer = f"达到步数上限 {state['max_steps']}，未能完成"
        return "end"
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):   # 还要调工具 → 去 act
        return "act"
    t.answer = last.content                 # 没有工具调用 = 它要给最终答案了
    t.answer_with_refs = ag.append_refs(t.answer, t.citations)
    return "end"


def build_graph():
    """★把节点和边装成图。这 6 行就是「编排」本身。"""
    g = StateGraph(AgentState)
    g.add_node("think", think)
    g.add_node("act", act)
    g.add_edge(START, "think")
    # think 之后分叉：要么去干活，要么结束
    g.add_conditional_edges("think", should_continue, {"act": "act", "end": END})
    g.add_edge("act", "think")      # ★干完活回去再想 —— 这条回边就是「循环」
    return g.compile()


GRAPH = build_graph()


def run(client, question, verbose=True, max_steps=ag.MAX_STEPS, log=True) -> ag.Trace:
    """★签名与手写版 ag.run 完全一致 —— 这样对比脚本能直接换着调，不用改调用方。"""
    t = ag.Trace(question=question)
    t.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    t.started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    # ★recursion_limit：图的硬闸。max_steps 是我的业务上限，这个是框架层的兜底。
    #   一轮 = think + act 两个节点，所以给 2 倍余量 + 2。
    GRAPH.invoke(
        {"messages": [{"role": "system", "content": ag.SYSTEM},
                      {"role": "user", "content": question}],
         "trace": t, "max_steps": max_steps, "verbose": verbose, "_client": client},
        config={"recursion_limit": max_steps * 2 + 2})

    t.elapsed = round(time.time() - t0, 2)
    if log:
        try:
            ag.save_log(t)
        except Exception as e:
            print(f"（日志写入失败，不影响回答：{e}）")
    return t


if __name__ == "__main__":
    print("★图结构：")
    print(GRAPH.get_graph().draw_ascii())
    q = sys.argv[1] if len(sys.argv) > 1 else "三次实验里哪次 loss 最低，训练了多少小时"
    print(f"\n★问：{q}\n")
    t = run(ag.new_client(), q)
    print(f"\n★答：{t.answer}")
    print(f"\n步数 {t.steps}｜工具 {t.tools_used}｜耗时 {t.elapsed}s｜"
          f"token {t.prompt_tokens + t.completion_tokens}｜降级 {t.degraded}")
