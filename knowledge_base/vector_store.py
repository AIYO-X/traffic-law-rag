# -*- coding: utf-8 -*-
"""
LanceDB 向量存储与语义检索
==========================
职责：
1. 将知识 chunk 批量嵌入后写入本地 LanceDB 表；
2. 基于余弦相似度执行 Top-K 语义检索（混合关键词加权）；
3. 实现多层检索优化：阈值过滤 → 同源限流 → 内容去重。

相似度计算说明：
- LanceDB 的 cosine metric 返回余弦距离 d = 1 - cos(θ)，范围约 [0, 2]；
- 本项目转换为相似度 score = 1 - d，范围 [-1, 1]，值越大越相关；
- 混合检索：最终得分 = 语义相似度 + KEYWORD_BOOST_WEIGHT × 关键词命中率；
  关键词命中率取问题字符 2-gram 在「标题 + 标签 + 分类」中的覆盖率（0~1），
  用于弥补中文短问句纯向量检索区分度不足的问题（如「闯红灯扣几分」精确命中
  「闯红灯的记分与认定方式」而非泛化的「记分周期」）；
- 使用 SIMILARITY_THRESHOLD 过滤弱相关结果，避免把无关知识塞进 Prompt；
- 多段融合策略：同一法规条目最多保留 MAX_PER_ITEM 个片段，防止单一来源霸榜；
- 内容去重：对字符 2-gram 计算 Jaccard 相似度，超过阈值时保留得分更高者。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import lancedb  # noqa: E402
except ImportError:  # 允许在未安装依赖时仍可导入模块（例如运行环境检测脚本）
    lancedb = None

from config import Config  # noqa: E402
from knowledge_base.loader import build_chunks, chunk_stats  # noqa: E402
from rag.embedder import OllamaEmbedder  # noqa: E402


class VectorStore:
    """本地向量知识库。"""

    def __init__(self, uri=None, table_name=None, embedder: OllamaEmbedder = None):
        self.uri = Path(uri) if uri else Path(Config.LANCEDB_URI)
        self.table_name = table_name or Config.TABLE_NAME
        self.embedder = embedder or OllamaEmbedder()
        self._db = None
        self._table = None

    # ---------- 连接管理 ----------
    def _connect(self):
        if lancedb is None:
            raise RuntimeError("未安装 lancedb，请先执行 `pip install -r requirements.txt`。")
        if self._db is None:
            self.uri.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.uri))
        return self._db

    def _get_table(self):
        if self._table is not None:
            return self._table
        db = self._connect()
        tables = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
        names = list(tables) if not hasattr(tables, "keys") else list(tables.keys())
        if self.table_name not in names:
            return None
        self._table = db.open_table(self.table_name)
        return self._table

    def is_ready(self) -> bool:
        """向量库是否已建好且有数据。"""
        try:
            table = self._get_table()
            return table is not None and table.count_rows() > 0
        except Exception:
            return False

    def count(self) -> int:
        table = self._get_table()
        return table.count_rows() if table is not None else 0

    # ---------- 建库 ----------
    def build(self, rebuild: bool = True, batch_size: int = 16) -> Dict[str, Any]:
        """构建向量库：加载知识 → 分段 → 嵌入 → 写入 LanceDB。"""
        chunks = build_chunks()
        if not chunks:
            raise RuntimeError("没有可用的知识片段，请检查知识库文件。")

        texts = [chunk["text"] for chunk in chunks]
        vectors: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            print(f"  正在嵌入 {start + 1}-{min(start + batch_size, len(texts))}/{len(texts)} ...")
            vectors.extend(self.embedder.embed(batch))

        if len(vectors) != len(chunks):
            raise RuntimeError("嵌入向量数量与知识片段数量不一致，建库中止。")

        records = []
        for chunk, vector in zip(chunks, vectors):
            row = dict(chunk)
            row["vector"] = vector
            records.append(row)

        db = self._connect()
        self._table = db.create_table(
            self.table_name,
            data=records,
            mode="overwrite" if rebuild else "append",
        )
        stats = chunk_stats(chunks)
        stats["written_rows"] = self._table.count_rows()
        return stats

    # ---------- 检索 ----------
    def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        max_per_item: int = 2,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """按向量执行混合检索，返回按融合得分降序排列的知识片段。

        混合检索 = 语义召回 + 关键词前置召回（取并集）：
        中文短问句的向量区分度不足，且跨领域提问（如「充电宝能带上飞机吗」）
        与正确条目的语义相似度可能很低（<0.55），仅靠向量召回会漏掉正确条目。
        因此在向量召回之外，额外按标题/标签/分类做关键词精确召回，把关键词
        强命中的条目强制并入候选集，再统一计算融合得分排序。
        """
        table = self._get_table()
        if table is None:
            raise RuntimeError("向量库尚未构建，请先运行 `python scripts/build_knowledge_base.py`。")

        top_k = top_k or Config.TOP_K
        threshold = Config.SIMILARITY_THRESHOLD if threshold is None else threshold

        # 1. 语义召回：多召回一些，保证候选充足
        vector_candidates = (
            table.search(query_vector)
            .metric("cosine")
            .limit(max(top_k * 10, 50))
            .to_list()
        )

        # 2. 关键词前置召回：把标题/标签/分类命中问题术语的条目强制并入候选集。
        #    即使其语义相似度低于阈值，也先保留，由融合得分统一排序。
        keyword_candidates = self._keyword_recall(query_text) if query_text else []

        # 合并去重（以 chunk_id 为键，向量候选优先保留语义分，关键词候补充漏网条目）
        merged: Dict[str, Dict[str, Any]] = {}
        for row in vector_candidates:
            merged[row["chunk_id"]] = row
        for row in keyword_candidates:
            if row["chunk_id"] not in merged:
                merged[row["chunk_id"]] = row

        scored: List[Dict[str, Any]] = []
        for row in merged.values():
            if "_distance" in row:
                semantic = 1.0 - float(row.get("_distance", 1.0))
            else:
                # 关键词前置召回进来的条目没有语义分，用其向量与问题向量计算余弦相似度
                semantic = self._cosine_similarity(query_vector, row.get("vector"))
            # 关键词前置召回的条目语义分可能低于阈值，仍允许进入评分，
            # 但若语义分过低且无关键词命中，则丢弃，避免无关噪声。
            boost = self._keyword_hit_rate(query_text, row) if query_text else 0.0
            if semantic < threshold and boost <= 0.0:
                continue
            final_score = semantic + Config.KEYWORD_BOOST_WEIGHT * boost
            scored.append(
                {
                    "chunk_id": row.get("chunk_id", ""),
                    "item_id": row.get("item_id", ""),
                    "title": row.get("title", ""),
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                    "tags": row.get("tags", []),
                    "content": row.get("content", ""),
                    "score": round(final_score, 4),
                    "semantic_score": round(semantic, 4),
                    "keyword_boost": round(boost, 4),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)

        # 优化 1：同源限流，避免同一条法规的多个分段占满结果
        limited: List[Dict[str, Any]] = []
        per_item: Dict[str, int] = {}
        for row in scored:
            used = per_item.get(row["item_id"], 0)
            if used >= max_per_item:
                continue
            per_item[row["item_id"]] = used + 1
            limited.append(row)

        # 优化 2：内容去重（2-gram Jaccard）
        deduplicated: List[Dict[str, Any]] = []
        for row in limited:
            if any(
                self._ngram_jaccard(row["content"], kept["content"]) >= Config.DEDUP_THRESHOLD
                for kept in deduplicated
            ):
                continue
            deduplicated.append(row)

        # 优化 3：二次精排（rerank）。初排取 top_k * 倍数条进入精排，
        # 综合语义分 + 关键词命中 + 标题精确命中重新排序，再截断回 top_k。
        if Config.RERANK_ENABLED and query_text:
            candidates = deduplicated[: max(top_k * Config.RERANK_CANDIDATE_FACTOR, top_k)]
            deduplicated = self._rerank(candidates, query_text)[:top_k]
        else:
            deduplicated = deduplicated[:top_k]

        return deduplicated

    def _rerank(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """二次精排：综合语义分、关键词命中率、标题命中信号重新排序。

        动机：初排的融合分已经综合了语义与关键词，但跨领域提问时正确条目
        的语义分普遍偏低（<0.55），仅靠关键词加权（0.60×命中率）有时不足以
        把它顶到 Top-1。精排额外引入「标题精确命中」这一最强主题信号——
        问题若与标题强命中，几乎可以确定就是目标条款，应被显著突出。

        精排得分 = 语义分 + KEYWORD_BOOST_WEIGHT×关键词命中率
                  + RERANK_TITLE_BOOST×标题命中率
        返回按精排得分降序的列表，字段与初排结果一致（score 更新为精排得分）。
        """
        reranked = []
        for row in candidates:
            keyword_boost = row.get("keyword_boost", 0.0)
            title_hit = self._title_hit_rate(query, row)
            final_score = (
                row.get("semantic_score", 0.0)
                + Config.KEYWORD_BOOST_WEIGHT * keyword_boost
                + Config.RERANK_TITLE_BOOST * title_hit
            )
            item = dict(row)
            item["score"] = round(final_score, 4)
            # 记录精排引入的标题命中信号，便于调参与问题排查
            item["title_hit"] = round(title_hit, 4)
            reranked.append(item)
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    def _keyword_recall(self, query: str) -> List[Dict[str, Any]]:
        """关键词前置召回：全量扫描知识库，返回标题/标签/分类命中问题术语的条目。

        知识库规模较小（百级），全量扫描代价可忽略。返回的 row 不带 _distance，
        由上层统一计算语义分（语义召回已覆盖的条目会优先保留其真实语义分）。
        """
        table = self._get_table()
        if table is None:
            return []
        # 全量读取表数据（不依赖 pandas），用 PyArrow 表转成 dict 列表
        arrow_tbl = table.to_arrow()
        column_names = arrow_tbl.column_names
        rows: List[Dict[str, Any]] = []
        for i in range(arrow_tbl.num_rows):
            row = {name: arrow_tbl.column(name)[i].as_py() for name in column_names}
            rows.append(row)
        matched: List[Dict[str, Any]] = []
        for row in rows:
            if self._keyword_hit_rate(query, row) > 0.0:
                matched.append(row)
        return matched

    def search_by_text(self, query: str, top_k: Optional[int] = None,
                       threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """问题文本 → 向量 → 混合检索 的便捷封装。"""
        vector = self.embedder.embed_one(query)
        return self.search(vector, top_k=top_k, threshold=threshold, query_text=query)

    @classmethod
    def _keyword_hit_rate(cls, query: Optional[str], row: Dict[str, Any]) -> float:
        """问题字符 2-gram 在标题/标签/分类中的覆盖率，作为关键词加权信号。

        标题命中权重 2 倍（标题高度概括条款主题），标签 1 倍，分类 0.5 倍。
        返回 0~1 的命中率；(query 为空或无 2-gram 时返回 0。
        """
        if not query:
            return 0.0
        q_grams = cls._ngrams(query, 2)
        if not q_grams:
            return 0.0

        title = str(row.get("title", "") or "")
        category = str(row.get("category", "") or "")
        tags = row.get("tags", []) or []
        text = title + " " + " ".join(str(t) for t in tags) + " " + category
        grams = cls._ngrams(text, 2)
        if not grams:
            return 0.0

        hit = len(q_grams & grams) / len(q_grams)
        # 标题单独命中时给予额外加权（标题是最强主题信号）
        title_grams = cls._ngrams(title, 2)
        title_hit = len(q_grams & title_grams) / len(q_grams) if title_grams else 0.0
        return min(1.0, hit * 0.75 + title_hit * 0.5)

    @classmethod
    def _title_hit_rate(cls, query: Optional[str], row: Dict[str, Any]) -> float:
        """问题字符 2-gram 在标题中的覆盖率（0~1），作为精排的最强主题信号。

        标题高度概括条款主题，问题若与标题强命中，说明该条目几乎就是
        用户要问的条款，应在精排阶段被显著突出。
        """
        if not query:
            return 0.0
        q_grams = cls._ngrams(query, 2)
        if not q_grams:
            return 0.0
        title = str(row.get("title", "") or "")
        title_grams = cls._ngrams(title, 2)
        if not title_grams:
            return 0.0
        return len(q_grams & title_grams) / len(q_grams)

    @staticmethod
    def _ngrams(text: str, n: int = 2) -> set:
        return {text[i:i + n] for i in range(max(len(text) - n + 1, 0))}

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """两个向量的余弦相似度（范围约 -1~1）。"""
        if a is None or b is None:
            return 0.0
        if not isinstance(a, (list, tuple)):
            a = list(a)
        if not isinstance(b, (list, tuple)):
            b = list(b)
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    @classmethod
    def _ngram_jaccard(cls, left: str, right: str) -> float:
        a, b = cls._ngrams(left), cls._ngrams(right)
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


if __name__ == "__main__":
    store = VectorStore()
    if not store.is_ready():
        print("向量库未构建，请先运行：python scripts/build_knowledge_base.py")
    else:
        print(f"向量库已就绪，共 {store.count()} 条片段。")
        results = store.search_by_text("闯红灯扣几分？")
        for item in results:
            print(f"  [{item['score']:.3f}] {item['title']} — {item['source']}")
