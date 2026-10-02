r"""
第3步：4 个工具
—— 每个工具都是「普通 Python 函数 + 一段说明书(JSON)」，没有任何魔法

★ 为什么是这 4 个（计划书第3步）：
  search_docs  主力，查笔记
  read_config  结构化查询，和语义检索形成对比
  calculate    ★故意加的，制造「该调A却调B」的机会，评测才有意义
  list_files   简单工具，测模型会不会滥用
"""
import json
from pathlib import Path

# ---- 数据源（第3步先用写死的，第4步 search_docs 换成真读 .md 文件）----
CONFIG = {"batch_size": 2, "learning_rate": 1e-4, "rank": 16, "alpha": 32,
          "epoch": 3, "max_samples": 1000, "quantization": "4bit NF4"}

# 三次实验记录。数字出自 train_results.json / trainer_state.json（8.17 已核实）
EXPERIMENTS = {
    "脏数据100条": {"train_loss": 0.6587, "时长": "47分钟", "数据量": 100},
    "脏数据全量": {"train_loss": 0.3547, "时长": "6小时27分", "数据量": 1197},
    "干净数据全量": {"train_loss": 0.2828, "时长": "6小时56分", "数据量": 1197,
                     "步数": 450, "训练秒数": 24987},
}

# 模拟知识库。第4步换成真读我的 .md 笔记 + 向量检索
DOCS = [
    {"来源": "每日理解记录.md", "内容": "4bit 量化推理输出乱码：NF4 量化后直接推理，"
     "输出出现重复字符和乱码。原因是量化配置和推理时的 dtype 不一致。"},
    {"来源": "每日理解记录.md", "内容": "三次训练实验的 train_loss：脏数据100条 0.6587，"
     "脏数据全量 0.3547，干净数据全量 0.2828。前一段降幅是数据量带来的，后一段是数据质量带来的。"},
    {"来源": "技术复盘记录.md", "内容": "数据质量排查：nuScenes 的 3D 标注是 360 度全景，"
     "与前置摄像头视角不符，导致目标数虚高。用 nuscenes-devkit 做 3D 到 2D 投影 + 40 米过滤后修复。"},
    {"来源": "eval_counting_result.json", "内容": "评测结果：10 场景 50 张图产出 139 个类别级比较项，"
     "严格准确率 0.3022，容错正负1 为 0.6547，MAE 1.6619。"},
    {"来源": "技术复盘记录.md", "内容": "分类别误差画像：卡车 47%、公交 30%、自行车 75% 数得准；"
     "轿车 23.9%、交通锥 15.4% 数不准。规律是密集小目标数不准。"},
]


# ==================== 4 个工具函数 ====================

LAST_CITATIONS = []   # ★第6步-①：search_docs 最近一次命中的出处，由 agent.py 取走


def search_docs(query: str, top_k: int = 5) -> str:
    # ★top_k 从 3 调到 5 的依据（9.2 诊断_乱码题.py 实测，不是拍脑袋）：
    #   问「4bit 乱码」时真答案排第 2-4 名，Top-3 会漏掉一部分。Top-5 能稳定覆盖。
    """★第4步已接真实知识库：读我自己的 3 份 md 笔记（64 片段）+ BGE 向量检索。
    ★命中为空时必须明说「没找到」，不许让模型自己编。"""
    import vec
    hits = vec.search(query, top_k=top_k, with_score=True)
    # 相似度过低 = 其实没相关内容，宁可说没有。0.35 是看实测分布定的（相关的都在 0.45+）
    hits = [(s, c) for s, c in hits if s >= 0.35]
    # ★第6步-① 引用溯源：把本次命中的出处记到全局，供 agent.py 收集进 Trace。
    #   用全局变量而不是改返回值，是为了不破坏「工具只返回字符串给模型」这个约定。
    LAST_CITATIONS.clear()
    LAST_CITATIONS.extend({"source": c.source, "heading": c.heading, "score": round(s, 3)}
                          for s, c in hits)
    if not hits:
        return "知识库里没有找到相关内容"
    return "\n\n".join(f"[来源：{c.source} → {c.heading}｜相似度{s:.2f}]\n{c.text}"
                       for s, c in hits)


def read_config(key: str) -> str:
    return str(CONFIG.get(key, f"没有 {key} 这个参数"))


def read_experiment(name: str = "") -> str:
    """查实验记录。不传 name 就返回全部（★③号题第一步靠这个比出哪次 loss 最低）"""
    if not name:
        return json.dumps(EXPERIMENTS, ensure_ascii=False)
    for k, v in EXPERIMENTS.items():
        if name in k or k in name:
            return json.dumps({k: v}, ensure_ascii=False)
    return f"没有叫 {name} 的实验，可选：{list(EXPERIMENTS)}"


def calculate(expr: str) -> str:
    """算数。★不用 eval —— eval 能执行任意代码，模型传什么就跑什么，是真实安全漏洞。
    改用 ast 只放行数字和四则运算。"""
    import ast
    import operator
    OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"不支持的表达式：{ast.dump(node)}")

    try:
        return str(round(ev(ast.parse(expr, mode="eval").body), 6))
    except Exception as e:
        return f"算不了：{e}"


def list_files(directory: str = ".") -> str:
    """列文件。简单工具，用来测模型会不会滥用它"""
    p = Path(directory)
    if not p.is_dir():
        return f"{directory} 不是一个目录"
    return "、".join(sorted(f.name for f in p.iterdir())[:20]) or "空目录"


# ==================== Tool Schema（说明书）====================
# ★8.24 实验结论：description 不决定「调不调」，决定「参数填得对不对」。
#   写糊了模型会瞎填参数(如 training.rank)，而且**静悄悄查空、不报错**。
#   所以这里每个都写明：能查什么、参数长什么样。

def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    _fn("search_docs",
        "在我的技术笔记里做语义检索，返回相关片段和来源文件名。适合查现象、原因、经验类问题，"
        "比如「4bit 为什么乱码」「误差画像是什么」。不能用来查具体配置参数的数值。",
        {"query": {"type": "string", "description": "检索关键词，如 4bit 乱码"}}, ["query"]),

    _fn("read_config",
        "查询本次微调训练的配置参数值。可查的键名只有这些："
        "batch_size、learning_rate、rank、alpha、epoch、max_samples、quantization。"
        "参数 key 必须严格用上面的键名原文，不要加前缀（不要写 training.rank）。",
        {"key": {"type": "string", "description": "键名原文，如 rank"}}, ["key"]),

    _fn("read_experiment",
        "查询三次训练实验的记录，含 train_loss、时长、数据量、步数。"
        "实验名只有三个：脏数据100条、脏数据全量、干净数据全量。"
        "★不确定该查哪次时，name 传空字符串可一次拿到全部三次记录来比较。",
        {"name": {"type": "string", "description": "实验名，或传空字符串取全部"}}, []),

    _fn("calculate",
        "做四则运算。只接受纯数学表达式，如 (0.3547-0.2828)/0.3547*100。"
        "不要传中文、不要传变量名。",
        {"expr": {"type": "string", "description": "数学表达式"}}, ["expr"]),

    _fn("list_files",
        "列出某个目录下的文件名。只在用户明确问「有哪些文件」时使用。",
        {"directory": {"type": "string", "description": "目录路径"}}, []),
]

DISPATCH = {"search_docs": search_docs, "read_config": read_config,
            "read_experiment": read_experiment, "calculate": calculate,
            "list_files": list_files}
