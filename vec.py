r"""
第4步后半段：向量检索（BGE-small-zh）

★核心思路（一句话）：
  把每段文字变成一串数字（向量），意思相近的两段，数字也相近。
  于是「找意思相近的段落」就变成了「找数字最接近的向量」——数学问题，可以算。

★为什么要换掉关键词检索（8.30 实测的真实理由，不是"因为更先进"）：
  问「严格准确率 是多少」时，关键词版把讲「显存要多少」的段落排第 1，
  因为「多少」这个无意义高频词刷了分。关键词只会数字面重合，不懂意思。
"""
import os
import sys
from pathlib import Path

# ★必须在 import transformers 之前设置：国内直连 huggingface.co 不通，走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

from kb import get_chunks

sys.stdout.reconfigure(encoding="utf-8")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"   # 约 100MB，本地跑，免费
CACHE = Path(__file__).parent / "vec_cache.npz"

_model = None
_vecs = None


def get_model():
    """首次调用会下载模型（约100MB），之后从本地缓存加载"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"（加载 {MODEL_NAME}，首次会下载约100MB…）")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build(force=False):
    """把 64 个片段全部转成向量，存到本地。★算一次就够，之后复用"""
    global _vecs
    chunks = get_chunks()
    if not force and CACHE.exists():
        d = np.load(CACHE)
        if d["vecs"].shape[0] == len(chunks):   # 片段数没变才敢复用缓存
            _vecs = d["vecs"]
            return _vecs
    texts = [f"{c.heading}。{c.text}" for c in chunks]   # 标题也喂进去，信息更全
    # normalize=True → 向量长度归一，之后点积就等于余弦相似度，省一步除法
    _vecs = get_model().encode(texts, normalize_embeddings=True,
                               show_progress_bar=True, batch_size=16)
    np.savez(CACHE, vecs=_vecs)
    return _vecs


def search(query: str, top_k=3, with_score=False):
    """向量检索：查询转向量 → 和 64 个片段算余弦相似度 → 取最高的 top_k

    ★BGE 官方要求：查询要加前缀 "为这个句子生成表示以用于检索相关文章："
      不加会掉点。这是模型训练时的约定，属于「用对工具」的细节。
    """
    vecs = build()
    q = get_model().encode(["为这个句子生成表示以用于检索相关文章：" + query],
                           normalize_embeddings=True)[0]
    sims = vecs @ q                      # 归一化过了，点积 = 余弦相似度
    idx = np.argsort(-sims)[:top_k]      # ★负号 = 降序（和 PCA 那个坑同一个手法）
    chunks = get_chunks()
    if with_score:
        return [(float(sims[i]), chunks[i]) for i in idx]
    return [chunks[i] for i in idx]
