r"""测 FastAPI 的四个接口都通不通（要先启动服务）

跑法（★开两个 cmd 窗口）：
  窗口1：python -m uvicorn api:app --host 127.0.0.1 --port 8000
  窗口2：python 测接口.py
"""
import json
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
BASE = "http://127.0.0.1:8000"
fails = []


def call(method, path, body=None, timeout=90):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


if __name__ == "__main__":
    try:
        st, h = call("GET", "/health")
    except urllib.error.URLError as e:
        print(f"★连不上服务：{e}\n  先在另一个窗口启动 uvicorn（见本文件顶部）")
        sys.exit(1)

    print("\n① GET /health")
    check("状态 ok", h.get("status") == "ok", str(h))

    print("\n② GET /tools")
    st, tools = call("GET", "/tools")
    names = [t["name"] for t in tools]
    check("返回 5 个工具", len(tools) == 5, str(names))
    check("含 search_docs", "search_docs" in names)

    print("\n③ POST /chat（查配置，应调 read_config 答 16）")
    st, r = call("POST", "/chat", {"question": "LoRA 的 rank 设的多少"})
    check("答案含 16", "16" in r["answer"], r["answer"][:60])
    check("用了 read_config", "read_config" in r["tools_used"], str(r["tools_used"]))
    check("有 run_id", bool(r["run_id"]), r["run_id"])

    print("\n④ POST /chat（检索题，要求返回决策链 + 出处）")
    st, r2 = call("POST", "/chat",
                  {"question": "4bit 推理为什么会输出乱码", "with_trace": True})
    check("有出处", len(r2["citations"]) > 0, f"{len(r2['citations'])} 条")
    check("出处带文件名", all("source" in c for c in r2["citations"]))
    check("决策链非空", bool(r2["trace"] and r2["trace"]["tool_calls"]),
          f"{len(r2['trace']['tool_calls'])} 次工具调用")

    print("\n⑤ 输入校验（★这些必须被拒）")
    for body, why in [({"question": ""}, "空问题"),
                      ({"question": "x" * 600}, "超长问题"),
                      ({"question": "测试", "max_steps": 999}, "max_steps 越界")]:
        try:
            call("POST", "/chat", body, timeout=20)
            check(f"拒绝{why}", False, "★居然放过了")
        except urllib.error.HTTPError as e:
            check(f"拒绝{why}", e.code == 422, f"HTTP {e.code}")

    print("\n⑥ GET /runs（可观测：日志能查）")
    st, runs = call("GET", "/runs?limit=5")
    check("有运行记录", len(runs) > 0, f"{len(runs)} 条")
    if runs:
        st, one = call("GET", f"/runs/{runs[0]['run_id']}")
        check("单次详情含决策链", "tool_calls" in one and "thoughts" in one)
    print("\n⑦ 路径穿越防护（★必须被拒）")
    # ★9.26 修正：原来只测 ..%2F..%2F.env，返回 404 —— 但那是**路由没匹配上**
    #   （%2F 解码成斜杠后路径变三段，{run_id} 吃不下），我的校验根本没执行。
    #   ★两条分开测，才能证明「校验本身」有效：
    for rid, want, why in [
        ("..%2F..%2F.env", 404, "带斜杠(路由层就挡了，走不到校验)"),
        ("a.b", 400, "带点号(★真正触发我的 isalnum 校验)"),
        ("x/y", 404, "带斜杠"),
    ]:
        try:
            call("GET", f"/runs/{rid}", timeout=20)
            check(f"拒绝 {rid}", False, "★居然放过了")
        except urllib.error.HTTPError as e:
            check(f"拒绝 {rid}", e.code == want, f"HTTP {e.code}（期望{want}）{why}")

    print(f"\n{'='*60}")
    print("★ 全部接口通过" if not fails else f"★★ {len(fails)} 项失败：{fails}")
    sys.exit(1 if fails else 0)
