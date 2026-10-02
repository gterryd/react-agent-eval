r"""第6步-③ FastAPI 服务化

★分层（对应 JD「接口设计、服务开发」）：
  api.py    ← 只管 HTTP：收请求、校验、转成响应。**不含任何业务逻辑**
  agent.py  ← ReAct 循环
  tools.py  ← 工具实现
  kb.py/vec.py ← 知识库与检索

⚠️⚠️ 安全边界（★必须主动说明，不装作没这回事）：
  **本服务没有任何鉴权、没有限流**，只绑 127.0.0.1 供本机演示。
  真要对外提供服务，至少要加：API Key/JWT 鉴权、按 IP 或用户限流、
  请求体大小上限、CORS 白名单、把 API key 挪到密钥管理服务。
  ★我没做这些，因为这是单用户的评测导向项目，没有真实流量。

启动（cmd，先 cd 到项目目录）：
  python -m uvicorn api:app --host 127.0.0.1 --port 8000
然后浏览器打开：http://127.0.0.1:8000/docs
"""
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import agent as ag
import tools as tl

sys.stdout.reconfigure(encoding="utf-8")

_client = None          # ★复用同一个 client，不要每个请求新建
_warmed = False


def get_client():
    global _client
    if _client is None:
        _client = ag.new_client()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """★启动预热：首次检索要加载 BGE 向量模型（约15s）。
    不预热的话第一个请求的人白等 15 秒 —— 9.24 看日志时实测到的冷启动开销。
    ★用 lifespan 而不是已废弃的 @app.on_event。"""
    global _warmed
    t0 = time.time()
    try:
        tl.search_docs("预热")
        _warmed = True
        print(f"[启动] 知识库与向量模型预热完成，耗时 {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[启动] 预热失败（不影响启动，首次请求会慢）：{e}")
    yield
    print("[关闭] 服务已停止")


app = FastAPI(
    title="VLM 项目问答 Agent",
    description="基于 ReAct 的多工具 Agent。⚠️本地演示服务，无鉴权无限流。",
    version="0.6.3",
    lifespan=lifespan,
)


class ChatIn(BaseModel):
    # ★用 pydantic 做输入校验：长度上限防超长请求，max_steps 夹在 1-8 防有人传 999 烧钱
    question: str = Field(..., min_length=1, max_length=500,
                          description="要问的问题")
    max_steps: int = Field(ag.MAX_STEPS, ge=1, le=8,
                           description="ReAct 最多走几步，防死循环")
    with_trace: bool = Field(False, description="是否返回每一步的决策链")


class ChatOut(BaseModel):
    answer: str
    answer_with_refs: str
    citations: list
    tools_used: list
    steps: int
    elapsed: float
    total_tokens: int
    hit_max_steps: bool
    run_id: str
    trace: dict | None = None


@app.get("/health", summary="健康检查")
def health():
    """★运维最基本的接口：服务活着吗、知识库装好了吗"""
    import kb
    return {"status": "ok", "预热完成": _warmed,
            "知识库片段数": len(kb.get_chunks()), "工具数": len(tl.TOOLS)}


@app.get("/tools", summary="列出 Agent 有哪些工具")
def list_tools():
    """★把工具清单暴露出来 —— 这就是模型看到的那份「说明书」。
    ★8.24 实测过：说明书写糊，模型会瞎填参数而且静悄悄查空。所以它值得被看见。"""
    return [{"name": f["function"]["name"],
             "description": f["function"]["description"],
             "parameters": list(f["function"]["parameters"]["properties"]),
             "required": f["function"]["parameters"].get("required", [])}
            for f in tl.TOOLS]


@app.post("/chat", response_model=ChatOut, summary="问一个问题")
def chat(req: ChatIn):
    try:
        t = ag.run(get_client(), req.question, verbose=False,
                   max_steps=req.max_steps, log=True)
    except Exception as e:
        # ★不把内部异常细节吐给调用方（可能含路径/密钥线索），只回一句话 + 打本地日志
        print(f"[/chat 失败] {type(e).__name__}: {e}")
        raise HTTPException(status_code=502, detail="Agent 执行失败，请查看服务端日志")

    out = ChatOut(
        answer=t.answer, answer_with_refs=t.answer_with_refs or t.answer,
        citations=t.citations, tools_used=t.tools_used, steps=t.steps,
        elapsed=t.elapsed, total_tokens=t.prompt_tokens + t.completion_tokens,
        hit_max_steps=t.hit_max_steps, run_id=t.run_id,
    )
    if req.with_trace:   # ★决策链默认不返回（很长），要看就显式要
        out.trace = {"thoughts": t.thoughts, "tool_calls": t.calls,
                     "llm_calls": t.llm_calls}
    return out


@app.get("/runs", summary="最近的运行记录（可观测）")
def recent_runs(limit: int = 20):
    """★把第6步-②的全链路日志开成接口 —— 日志能查才叫可观测"""
    import json
    f = ag.LOG_DIR / "index.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]


@app.get("/runs/{run_id}", summary="某次运行的完整决策链")
def one_run(run_id: str):
    import json
    # ★防路径穿越：run_id 只允许数字/字母/下划线，否则有人传 ../../.env 就能读走密钥
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="run_id 格式不合法")
    f = ag.LOG_DIR / f"{run_id}.json"
    if not f.exists():
        raise HTTPException(status_code=404, detail="没有这条运行记录")
    return json.loads(f.read_text(encoding="utf-8"))
