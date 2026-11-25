# -*- coding: utf-8 -*-
"""
向量知识库构建脚本
==================
流程：检查 Ollama → 加载知识条目 → 分段 → 批量嵌入 → 写入 LanceDB → 检索自测。

用法：
    python scripts/build_knowledge_base.py
    python scripts/build_knowledge_base.py --query "醉酒驾驶怎么处罚？"
    python scripts/build_knowledge_base.py --threshold 0.2   # 放宽检索阈值
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from knowledge_base.loader import load_knowledge_items  # noqa: E402
from knowledge_base.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="构建交通法规向量知识库")
    parser.add_argument("--query", default="闯红灯扣几分？", help="建库后用于自测的问题")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖相似度阈值")
    parser.add_argument("--no-rebuild", action="store_true", help="追加而非覆盖已有表")
    args = parser.parse_args()

    print("=" * 62)
    print("交通法规向量知识库构建")
    print("=" * 62)

    items = load_knowledge_items()
    print(f"[1/3] 已加载 {len(items)} 条法规知识。")

    store = VectorStore()
    available, message = store.embedder.check_health()
    print(f"[2/3] {message}")
    if not available:
        print("\n构建中止：请先启动 Ollama 并拉取嵌入模型后重试。")
        return 1

    print("[3/3] 开始嵌入并写入 LanceDB…")
    stats = store.build(rebuild=not args.no_rebuild)
    print("\n构建统计：")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    threshold = args.threshold if args.threshold is not None else Config.SIMILARITY_THRESHOLD
    print(f"\n检索自测（问题：{args.query}，阈值：{threshold}）：")
    results = store.search_by_text(args.query, threshold=threshold)
    if not results:
        print("  未命中任何片段，可尝试降低 --threshold。")
    for index, item in enumerate(results, 1):
        print(f"  [{index}] 相似度 {item['score']:.3f} | {item['title']}")
        print(f"       出处：{item['source']}")

    print("\n向量库构建完成。现在可以运行：python app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
