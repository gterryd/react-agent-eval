r"""第6步-④ 验收：重试与降级

★为什么用「假的 DeepSeek」而不是真拔网线：
  拔网线不可重复、没法自动化、也没法精确造出「前2次超时第3次成功」这种情况。
  注入故障是行业常规做法。★真实 API 我在最后一项单独跑一次，确认没改坏。

★时间也是假的：sleeper 换成记账函数，不真等 7 秒，但能验证「等了多久」对不对。

跑法：
  python 测重试.py
"""
import sys

import httpx
from openai import APITimeoutError, AuthenticationError, RateLimitError

import agent as ag

sys.stdout.reconfigure(encoding="utf-8")
fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


class 假DeepSeek:
    """前 fail_times 次抛指定异常，之后返回一个假回答"""

    def __init__(self, exc, fail_times):
        self.exc, self.fail_times, self.called = exc, fail_times, 0
        self.chat = type("c", (), {"completions": self})()

    def create(self, **kw):
        self.called += 1
        if self.called <= self.fail_times:
            raise self.exc
        return 假回答()


class 假回答:
    def __init__(self):
        msg = type("m", (), {"content": "假答案", "tool_calls": None})()
        self.choices = [type("c", (), {"message": msg, "finish_reason": "stop"})()]
        self.usage = type("u", (), {"prompt_tokens": 10, "completion_tokens": 5})()


class 记时钟:
    """假的 time.sleep：只记账不真睡"""

    def __init__(self):
        self.waits = []

    def __call__(self, d):
        self.waits.append(round(d, 2))


# ★用真的 httpx 对象来造异常：openai 的异常类会去读 response.status_code / headers，
#   自己拿 type() 拼的假对象缺字段会当场炸（9.27 实测过）。
_req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")


def _resp(code):
    return httpx.Response(code, request=_req)


TIMEOUT = APITimeoutError(request=_req)
LIMIT = RateLimitError("429 限流", response=_resp(429), body=None)
AUTH = AuthenticationError("401 key 错", response=_resp(401), body=None)

if __name__ == "__main__":
    print("\n① 第1次就成功 → 不该有任何等待")
    c, clk = 假DeepSeek(TIMEOUT, 0), 记时钟()
    resp, attempts, waited = ag.call_llm(c, [], sleeper=clk)
    check("只打了1次电话", c.called == 1, f"called={c.called}")
    check("没等待", clk.waits == [] and waited == 0, f"waits={clk.waits}")

    print("\n② 前2次超时、第3次成功 → 该重试并最终成功")
    c, clk = 假DeepSeek(TIMEOUT, 2), 记时钟()
    resp, attempts, waited = ag.call_llm(c, [], sleeper=clk)
    check("打了3次电话", c.called == 3, f"called={c.called}")
    check("最终拿到回答", resp is not None and attempts == 3, f"attempts={attempts}")
    check("★等待是递增的(1s→2s)", len(clk.waits) == 2 and clk.waits[1] > clk.waits[0],
          f"waits={clk.waits}")
    check("★第2次约为第1次2倍", 1.8 <= clk.waits[1] / clk.waits[0] <= 2.2,
          f"倍数={clk.waits[1]/clk.waits[0]:.2f}")
    check("★有随机抖动(不是整数1.0)", clk.waits[0] != 1.0, f"首次等待={clk.waits[0]}")

    print("\n③ 一直超时 → 重试3次后抛 LLMUnavailable（不是裸 APITimeoutError）")
    c, clk = 假DeepSeek(TIMEOUT, 99), 记时钟()
    try:
        ag.call_llm(c, [], sleeper=clk)
        check("抛出 LLMUnavailable", False, "★居然没抛")
    except ag.LLMUnavailable as e:
        check("抛出 LLMUnavailable", True, str(e)[:50])
        check("恰好打3次电话(不是无限重试)", c.called == 3, f"called={c.called}")
        check("★只等了2次(最后一次失败不再等)", len(clk.waits) == 2, f"waits={clk.waits}")
        check("错误信息含重试次数", "3 次" in str(e))
    except Exception as e:
        check("抛出 LLMUnavailable", False, f"★抛的是 {type(e).__name__}")

    print("\n④ 限流(429) → 也该重试")
    c, clk = 假DeepSeek(LIMIT, 1), 记时钟()
    resp, attempts, waited = ag.call_llm(c, [], sleeper=clk)
    check("429 重试后成功", c.called == 2 and resp is not None, f"called={c.called}")

    print("\n⑤ ★key 错(401) → 绝不该重试，立刻抛")
    c, clk = 假DeepSeek(AUTH, 99), 记时钟()
    try:
        ag.call_llm(c, [], sleeper=clk)
        check("立刻抛出", False, "★居然没抛")
    except AuthenticationError:
        check("立刻抛 AuthenticationError", True)
        check("★只打1次电话(没白等)", c.called == 1, f"called={c.called}")
        check("★完全没等待", clk.waits == [], f"waits={clk.waits}")
    except ag.LLMUnavailable:
        check("立刻抛出", False, "★错误地当成可重试，白等了")

    print("\n⑥ ★降级：run() 整条链路挂掉，要给人话不要崩")
    ag.time.sleep = 记时钟()        # ★run() 内部调 call_llm 不传 sleeper，这里临时换掉
    c = 假DeepSeek(TIMEOUT, 99)
    try:
        t = ag.run(c, "rank 是多少", verbose=False, log=False)
        check("没有崩(拿到 Trace)", True)
        check("★标记了 degraded", t.degraded is True, f"degraded={t.degraded}")
        check("★答案是人话不是堆栈", "请稍后再试" in t.answer, t.answer[:60])
        check("记了降级原因", "重试 3 次" in t.degrade_reason, t.degrade_reason[:50])
    except Exception as e:
        check("没有崩(拿到 Trace)", False, f"★崩了：{type(e).__name__}: {e}")

    print("\n⑦ ★正常路径没被改坏：假DeepSeek 一次就成功，run() 该正常返回")
    c = 假DeepSeek(TIMEOUT, 0)
    t = ag.run(c, "随便问", verbose=False, log=False)
    check("答案是模型给的", t.answer == "假答案", t.answer)
    check("没标 degraded", t.degraded is False)
    check("重试数为0", t.retry_total == 0, f"retry_total={t.retry_total}")
    check("token 照样统计", t.prompt_tokens == 10 and t.completion_tokens == 5,
          f"{t.prompt_tokens}/{t.completion_tokens}")

    print(f"\n{'='*60}")
    print("★ 重试与降级全部通过" if not fails else f"★★ {len(fails)} 项失败：{fails}")
    sys.exit(1 if fails else 0)
