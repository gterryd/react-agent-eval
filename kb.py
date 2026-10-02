r"""
第4步（前半段）：把真实的 .md 笔记切成可检索的片段

★为什么要切片：整份文档几千字，一次全塞给模型太贵也太吵。
  切成小片段，只把相关的几片给它。

★切法：按 markdown 标题切（# ## ###）。
  理由：标题天然是「一个话题」的边界，比按固定字数切更不容易切断一句话。
  ——这是最朴素的切法，计划里明确不做切分策略对比实验（那是 RAG 专项优化）。
"""
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

def _load_env(path=Path(__file__).parent / ".env"):
    """★自己读 .env，不依赖调用方先调 agent.load_env() ——
    因为本模块在**被 import 的那一刻**就要读 KB_ROOT，晚了就读不到。
    （★10.2 踩过同类坑：HF_ENDPOINT 设晚了不生效。）"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

# ★知识库位置与文件清单都从环境变量读（配置示例见 .env.example）。
#   ★为什么不写死：知识库是作者的私人学习笔记，**没有入库**（见 README）。
#   任何人把自己的 markdown 笔记放一个目录、设好这两个变量就能跑。
#   ★一个都读不到时会响亮报错并给出配置方法，不会静悄悄返回空结果。
ROOT = Path(os.environ.get("KB_ROOT", "./notes"))

# KB_FILES：分号隔开的相对路径，如 "a.md;sub/b.md"
SOURCES = [s.strip() for s in os.environ.get("KB_FILES", "").split(";") if s.strip()]

MIN_CHARS = 30    # 太短的片段（只有个标题）没信息量，丢掉
MAX_CHARS = 800   # 太长的切开，否则一片就把上下文吃满


@dataclass
class Chunk:
    source: str    # 哪个文件 ★引用溯源靠这个
    heading: str   # 所属标题
    text: str

    def __str__(self):
        return f"[来源：{self.source} → {self.heading}]\n{self.text}"


def split_long(text, limit=MAX_CHARS):
    """超长片段按空行再切，尽量不切断句子"""
    if len(text) <= limit:
        return [text]
    out, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) > limit and buf:
            out.append(buf.strip())
            buf = para
        else:
            buf += "\n\n" + para
    if buf.strip():
        out.append(buf.strip())
    return out


def load_chunks(sources=SOURCES) -> list:
    """读 md → 按标题切 → 返回 Chunk 列表"""
    chunks = []
    for rel in sources:
        path = ROOT / rel
        if not path.exists():
            print(f"⚠️ 跳过不存在的文件：{rel}")
            continue
        heading = "（开头）"
        buf = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if re.match(r"^#{1,4}\s+", line):        # 遇到标题 = 上一段结束
                _flush(chunks, path.name, heading, buf)
                heading = line.lstrip("#").strip()
                buf = []
            else:
                buf.append(line)
        _flush(chunks, path.name, heading, buf)
    return chunks


def _flush(chunks, source, heading, buf):
    text = "\n".join(buf).strip()
    if len(text) < MIN_CHARS:
        return
    for piece in split_long(text):
        chunks.append(Chunk(source=source, heading=heading, text=piece))


_CACHE = None


def get_chunks():
    """只切一次，之后复用"""
    global _CACHE
    if _CACHE is None:
        _CACHE = load_chunks()
        # ★一个片段都没切到时必须响亮报错，不许静悄悄返回空列表 —— 否则检索永远查不到，
        #   而模型会说「知识库里没有」，看起来像 Agent 的问题，其实是知识库没装上。
        #   ★这正是 8.24「静悄悄查空」那个坑的同类，必须明说。
        if not _CACHE:
            raise FileNotFoundError(
                f"知识库是空的：在 {ROOT} 下没读到任何 .md 片段。\n"
                f"  本项目的知识库是作者私人学习笔记，未入库（见 README）。\n"
                f"  请把你自己的 markdown 笔记放一个目录，然后设置环境变量：\n"
                f"    set KB_ROOT=你的笔记目录\n"
                f"    set KB_FILES=笔记1.md;子目录\\笔记2.md\n"
                f"  当前读取清单：{SOURCES}")
    return _CACHE


def search(query: str, top_k=3) -> list:
    """★第4步前半段：关键词打分检索（后半段换成 BGE 向量检索）

    打分规则：查询里的每个词，在片段里出现就加分。标题命中额外加权。
    ★这是最土的办法，故意先做它 —— 后半段换向量后可以对比「土办法 vs 向量」差多少。
    """
    words = [w for w in re.split(r"[\s,，、。？?的是了]+", query) if len(w) >= 2]
    scored = []
    for c in get_chunks():
        score = sum(c.text.count(w) for w in words) + 3 * sum(w in c.heading for w in words)
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


if __name__ == "__main__":
    cs = get_chunks()
    print(f"★ 切片完成：{len(cs)} 个片段，来自 {len(set(c.source for c in cs))} 个文件\n")
    for src in sorted(set(c.source for c in cs)):
        sub = [c for c in cs if c.source == src]
        avg = sum(len(c.text) for c in sub) // len(sub)
        print(f"  {src:<28} {len(sub):>3} 片   平均 {avg} 字   "
              f"最长 {max(len(c.text) for c in sub)} 字")

    print(f"\n{'='*70}\n★ 试检索三个真问题（看它从我真实笔记里捞出什么）")
    for q in ["严格准确率 是多少", "交通锥 准确率", "3D 标注 投影 过滤"]:
        hits = search(q, top_k=2)
        print(f"\n问：{q}   → 命中 {len(hits)} 片")
        for h in hits:
            print(f"  [{h.source} → {h.heading}]")
            print(f"    {h.text[:110].replace(chr(10), ' ')}...")
