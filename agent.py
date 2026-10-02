r"""
第3步：ReAct 循环
—— ReAct = Reason(思考) + Act(行动)，循环着来：想 → 做 → 看结果 → 再想 → ... → 答

整个 Agent 的核心只有下面 run() 那 20 行。不神秘：
  就是「反复问模型：你还要调工具吗？」要 → 执行 → 把结果塞回去再问；不要 → 它开始回答了
"""
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import (APIConnectionError, APITimeoutError, AuthenticationError,
                    InternalServerError, OpenAI, RateLimitError)

from tools import DISPATCH, TOOLS

sys.stdout.reconfigure(encoding="utf-8")

MAX_STEPS = 6  # ★不设上限模型可能反复调同一个工具，死循环烧钱

# ========== 第6步-④ 错误处理与降级 ==========
MAX_RETRY = 3       # 最多打 3 次电话（1次原始 + 2次重试）
BASE_DELAY = 1.0    # 退避基数：等 1s → 2s → 4s
LLM_TIMEOUT = 60    # ★单次调用超时上限，不设的话卡住能等到天荒地老

# ★该重试的：对方的临时毛病，过会儿可能就好了
RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
# ★不该重试的：我自己的错，重试一万次还是错，只是白等
#   AuthenticationError(key错) / BadRequestError(参数非法) 都属于这类，直接抛


class LLMUnavailable(Exception):
    """重试完仍然不通 —— 用于降级，不让异常裸奔到用户面前"""


def call_llm(client, messages, retries=MAX_RETRY, base=BASE_DELAY, sleeper=time.sleep):
    """带指数退避重试的 API 调用。★sleeper 可注入，测试时不用真等 7 秒。

    返回 (resp, attempts, waited)：attempts=打了几次电话，waited=总共等了多久
    """
    waited, last = 0.0, None
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(
                model="deepseek-chat", messages=messages, tools=TOOLS,
                temperature=0, timeout=LLM_TIMEOUT), attempt, round(waited, 2)
        except RETRYABLE as e:
            last = e
            if attempt == retries:      # 最后一次也挂了，不用再等
                break
            # ★指数退避 + 随机抖动：1s→2s→4s，抖动防所有客户端同时重试撞一起
            delay = base * (2 ** (attempt - 1)) * (1 + random.random() * 0.1)
            print(f"（第{attempt}次调用失败：{type(e).__name__}，等 {delay:.1f}s 重试）")
            sleeper(delay)
            waited += delay
    raise LLMUnavailable(
        f"模型服务不可用，已重试 {retries} 次｜最后一次错误：{type(last).__name__}: {last}")


def load_env(path=Path(__file__).parent / ".env"):
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def new_client():
    load_env()
    return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                  base_url=os.environ["DEEPSEEK_BASE_URL"])


SYSTEM = (
    "你是一个只回答「我自己的 VLM 微调项目」相关问题的助手。\n"
    "规则：\n"
    "1. 涉及项目的事实（参数、loss、时长、评测数字）必须调工具查证，不许凭记忆回答。\n"
    "2. 需要算数就调 calculate，不要自己心算。\n"
    "3. 工具查不到、或问题与本项目无关（如天气、新闻），"
    "直接回答「知识库里没有这个信息」并说明原因，不要编。"
)


@dataclass
class Trace:
    """★一次运行的全过程记录。第5步算指标、第6步做可观测，全靠这个"""
    question: str
    tools_used: list = field(default_factory=list)   # 工具名，按顺序
    calls: list = field(default_factory=list)        # 完整调用明细
    answer: str = ""
    steps: int = 0
    elapsed: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    hit_max_steps: bool = False
    # ★第6步-① 引用溯源：这次回答实际用到了哪些文档片段（文件名 → 标题 → 相似度）
    citations: list = field(default_factory=list)
    # ★answer 保持纯净（评测判定用它，加了出处会干扰关键词匹配）；
    #   answer_with_refs = 带出处的版本，给人看、给界面展示用
    answer_with_refs: str = ""
    # ★第6步-② 全链路日志：每一步的完整现场
    #   ReAct 的 Reason(思考) 一直没被记下来 —— 只记了 Act(行动)，排查时看不到它"为什么"这么做
    thoughts: list = field(default_factory=list)   # 每步模型输出的思考文字
    llm_calls: list = field(default_factory=list)  # 每次 API 调用的耗时和 token
    run_id: str = ""
    started_at: str = ""
    # ★第6步-④：降级相关
    degraded: bool = False          # 是否走了降级（模型服务不可用）
    degrade_reason: str = ""
    retry_total: int = 0            # 本次问答一共重试了几次


LOG_DIR = Path(__file__).parent / "logs"


def save_log(t: "Trace") -> Path:
    """★第6步-② 全链路日志落盘。

    两份一起写，各有用途：
      logs/<run_id>.json  单次完整现场（排查用，看得到每一步）
      logs/index.jsonl    一行一次，只存摘要（统计用，不必读几百个大文件）
    ★这就是 JD 里的「Agent 可观测 / 全链路追踪」。RAG 只需记一次检索，
      Agent 要记整条决策链 —— 这是 Agent 项目独有的工作量。
    """
    LOG_DIR.mkdir(exist_ok=True)
    detail = {
        "run_id": t.run_id, "started_at": t.started_at, "question": t.question,
        "answer": t.answer, "answer_with_refs": t.answer_with_refs,
        "steps": t.steps, "总耗时": t.elapsed,
        "tools_used": t.tools_used, "hit_max_steps": t.hit_max_steps,
        "degraded": t.degraded, "degrade_reason": t.degrade_reason,
        "retry_total": t.retry_total,
        "tokens": {"prompt": t.prompt_tokens, "completion": t.completion_tokens,
                   "total": t.prompt_tokens + t.completion_tokens},
        "thoughts": t.thoughts,       # 每步在想什么（Reason）
        "tool_calls": t.calls,        # 每步做了什么（Act）：工具/参数/结果/耗时
        "llm_calls": t.llm_calls,     # 每次调模型的耗时和 token
        "citations": t.citations,
    }
    f = LOG_DIR / f"{t.run_id}.json"
    f.write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")

    brief = {k: detail[k] for k in ("run_id", "started_at", "question", "steps",
                                    "总耗时", "tools_used", "hit_max_steps",
                                    "degraded", "retry_total")}
    brief["total_tokens"] = detail["tokens"]["total"]
    brief["llm耗时合计"] = round(sum(c["llm耗时"] for c in t.llm_calls), 3)
    brief["工具耗时合计"] = round(sum(c["耗时"] for c in t.calls), 3)
    with (LOG_DIR / "index.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(brief, ensure_ascii=False) + "\n")
    return f


def append_refs(answer: str, citations: list) -> str:
    """★第6步-① 引用溯源：把出处拼到回答末尾，人可以去原文核对。

    ★为什么这件事重要：被问到「你怎么知道它没瞎说」，这就是答案 ——
      每句结论都能追到「哪个文件、哪一节」，可验证而不是靠信任。
    """
    if not citations:
        return answer
    lines = [f"  [{i}] {c['source']} → {c['heading']}（相似度 {c['score']}）"
             for i, c in enumerate(sorted(citations, key=lambda x: -x["score"]), 1)]
    return answer + "\n\n— 出处（可去原文核对）—\n" + "\n".join(lines)


def run(client, question, verbose=True, max_steps=MAX_STEPS, log=True) -> Trace:
    """★ReAct 循环本体。整个 Agent 就是这个 while"""
    t = Trace(question=question)
    t.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    t.started_at = datetime.now().isoformat(timespec="seconds")
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    t0 = time.time()

    for step in range(1, max_steps + 1):
        c0 = time.time()
        try:
            resp, attempts, waited = call_llm(client, messages)
        except LLMUnavailable as e:
            # ★降级：不抛异常给用户，给一句明确的话 + 把失败现场记进日志
            t.degraded = True
            t.degrade_reason = str(e)
            t.answer = f"抱歉，{e}。请稍后再试。"
            t.answer_with_refs = t.answer
            t.llm_calls.append({"step": step, "llm耗时": round(time.time() - c0, 3),
                                "失败": str(e)})
            break
        llm_cost = time.time() - c0
        msg = resp.choices[0].message
        t.prompt_tokens += resp.usage.prompt_tokens
        t.completion_tokens += resp.usage.completion_tokens
        # ★第6步-②：把这次 API 调用的现场记下来。★区分「等模型」和「跑工具」两种耗时，
        #   否则看到「这步花了3秒」也不知道该优化哪边
        t.retry_total += attempts - 1    # ★第6步-④：重试次数进日志，否则重试静悄悄发生没人知道
        t.llm_calls.append({"step": step, "llm耗时": round(llm_cost, 3),
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "finish_reason": resp.choices[0].finish_reason,
                            "调用次数": attempts, "重试等待": waited})
        if msg.content:   # ★ReAct 的 Reason 部分，以前没记，排查时看不到它为什么这么做
            t.thoughts.append({"step": step, "思考": msg.content})
        messages.append(msg)
        t.steps = step

        if not msg.tool_calls:          # 没有工具调用 = 它要给最终答案了
            t.answer = msg.content
            t.answer_with_refs = append_refs(t.answer, t.citations)
            break

        for tc in msg.tool_calls:       # 一步里可能并行调多个
            args = json.loads(tc.function.arguments or "{}")
            r0 = time.time()
            try:
                result = DISPATCH[tc.function.name](**args)
            except Exception as e:      # ★工具挂了不能让整个 Agent 崩，要能继续
                result = f"工具执行失败：{e}"
            cost = time.time() - r0
            # ★第6步-①：检索工具刚跑完，把它命中的出处收进 Trace（去重，保留最高分）
            if tc.function.name == "search_docs":
                import tools as _t
                for c in _t.LAST_CITATIONS:
                    same = next((x for x in t.citations
                                 if x["source"] == c["source"]
                                 and x["heading"] == c["heading"]), None)
                    if same is None:
                        t.citations.append(dict(c))
                    elif c["score"] > same["score"]:
                        same["score"] = c["score"]
            t.tools_used.append(tc.function.name)
            t.calls.append({"step": step, "tool": tc.function.name,
                            "args": args, "result": result, "耗时": round(cost, 3)})
            if verbose:
                print(f"  第{step}步 → {tc.function.name}({args})")
                print(f"         得到：{str(result)[:100]}")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": str(result)})
    else:
        t.hit_max_steps = True          # for 正常跑完没 break = 撞上限了
        t.answer = f"达到步数上限 {max_steps}，未能完成"

    t.elapsed = round(time.time() - t0, 2)
    if log:
        try:
            save_log(t)
        except Exception as e:      # ★写日志失败不能拖垮主流程
            print(f"（日志写入失败，不影响回答：{e}）")
    return t
