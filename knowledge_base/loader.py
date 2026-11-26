# -*- coding: utf-8 -*-
"""
知识库加载与分段（chunk）模块
=============================
职责：
1. 从 JSON 文件读取原始知识条目并校验字段完整性；
2. 将过长的知识片段按句子边界切分为带重叠的 chunk，避免语义被切断；
3. 为每个 chunk 保留标题、出处、分类、标签等溯源信息。

分段策略说明：
- 以「条」为基本语义单元：每条法规知识本身就是一个完整知识点；
- 仅当单条内容超过 CHUNK_SIZE 时才二次切分；
- 切分优先在句号、问号、感叹号、分号、换行处断开，保证语义完整；
- 相邻 chunk 保留 CHUNK_OVERLAP 个字符的重叠，避免边界信息丢失；
- 嵌入文本 = 标题 + 正文，使向量同时携带主题信号与细节信号。
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许以 `python knowledge_base/loader.py` 方式直接运行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402

REQUIRED_FIELDS = ("id", "category", "title", "content", "source")


def load_knowledge_items(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """加载原始知识条目，并校验必需字段。"""
    file_path = Path(path) if path else Config.KNOWLEDGE_FILE
    if not file_path.exists():
        raise FileNotFoundError(
            f"知识库文件不存在：{file_path}\n请先检查 data/ 目录，或运行 scripts/build_knowledge_base.py。"
        )

    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    items = data.get("items", []) if isinstance(data, dict) else data
    if not items:
        raise ValueError(f"知识库文件为空或缺少 items 字段：{file_path}")

    for index, item in enumerate(items, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in item or not item[field]]
        if missing:
            raise ValueError(f"第 {index} 条知识缺少字段 {missing}，请检查数据源。")
        if not isinstance(item.get("tags", []), list):
            raise ValueError(f"第 {index} 条知识的 tags 必须是列表。")

    return items


def split_sentences(text: str) -> List[str]:
    """按中文句读符号切分句子，保留分隔符。"""
    return [s.strip() for s in re.split(r"(?<=[。！？!?；;\n])", text) if s.strip()]


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    min_chars: Optional[int] = None,
) -> List[str]:
    """把长文本切分为带重叠的语义片段。"""
    chunk_size = chunk_size or Config.CHUNK_SIZE
    overlap = Config.CHUNK_OVERLAP if overlap is None else overlap
    min_chars = min_chars or Config.MIN_CHUNK_CHARS

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则会导致切分死循环。")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    sentences = split_sentences(text)
    chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        # 极端情况：单句长于 chunk_size，按字符硬切
        while len(sentence) > chunk_size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = buffer[-overlap:] if overlap else ""
            room = chunk_size - len(buffer)
            buffer += sentence[:room]
            sentence = sentence[room:]

        if len(buffer) + len(sentence) <= chunk_size:
            buffer += sentence
        else:
            chunks.append(buffer.strip())
            buffer = (buffer[-overlap:] if overlap else "") + sentence

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if len(c) >= min_chars]


def build_chunks(
    items: Optional[List[Dict[str, Any]]] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """把知识条目转换为可直接向量化的 chunk 列表。"""
    source_items = items if items is not None else load_knowledge_items()
    chunks: List[Dict[str, Any]] = []
    seen_ids = set()

    for item in source_items:
        for field in REQUIRED_FIELDS:
            if field not in item or not item[field]:
                raise ValueError(f"条目缺少必需字段 {field}：{item.get('id', '未知')}")

        item_id = str(item["id"])
        if item_id in seen_ids:
            continue  # 防止重复入库
        seen_ids.add(item_id)

        pieces = chunk_text(item["content"], chunk_size, overlap)
        for idx, piece in enumerate(pieces, 1):
            title = str(item["title"]).strip()
            chunks.append(
                {
                    "chunk_id": f"{item_id}-C{idx}",
                    "item_id": item_id,
                    "title": title,
                    "category": str(item["category"]),
                    "source": str(item["source"]),
                    "tags": list(item.get("tags", [])),
                    "content": piece,
                    # 嵌入文本携带标题信息，提升语义命中率
                    "text": f"【{title}】{piece}",
                    "chunk_index": idx,
                    "char_count": len(piece),
                }
            )

    return chunks


def chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """统计分段结果，供建库脚本使用。"""
    if not chunks:
        return {"chunk_count": 0}
    lengths = [c["char_count"] for c in chunks]
    categories: Dict[str, int] = {}
    for chunk in chunks:
        categories[chunk["category"]] = categories.get(chunk["category"], 0) + 1
    return {
        "chunk_count": len(chunks),
        "unique_items": len({c["item_id"] for c in chunks}),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "category_distribution": categories,
    }


if __name__ == "__main__":
    items = load_knowledge_items()
    result = build_chunks(items)
    print(f"原始知识条目：{len(items)} 条")
    print("分段统计：", json.dumps(chunk_stats(result), ensure_ascii=False, indent=2))
