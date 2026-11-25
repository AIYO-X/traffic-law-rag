# -*- coding: utf-8 -*-
"""
模型拉取脚本
============
封装 `ollama pull`，一键下载嵌入模型与生成模型。

用法：
    python scripts/pull_models.py                    # 拉取默认两个模型
    python scripts/pull_models.py --llm qwen2.5:7b   # 指定生成模型
    python scripts/pull_models.py --skip-llm         # 只拉嵌入模型
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402


def pull(model: str) -> bool:
    print(f"\n>>> 正在拉取模型：{model}")
    print("    首次下载可能较大，请耐心等待；可随时 Ctrl+C 中断。")
    try:
        result = subprocess.run(["ollama", "pull", model], check=False)
        if result.returncode == 0:
            print(f"<<< 模型 {model} 拉取完成。")
            return True
        print(f"<<< 模型 {model} 拉取失败，返回码 {result.returncode}。", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("未找到 ollama 命令。请先安装 Ollama：https://ollama.com/download", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="拉取 RAG 系统所需的 Ollama 模型")
    parser.add_argument("--embed", default=Config.EMBED_MODEL, help="嵌入模型名称")
    parser.add_argument("--llm", default=Config.LLM_MODEL, help="生成模型名称")
    parser.add_argument("--skip-embed", action="store_true", help="跳过嵌入模型")
    parser.add_argument("--skip-llm", action="store_true", help="跳过生成模型")
    args = parser.parse_args()

    if not shutil.which("ollama"):
        print("未检测到 ollama 命令。请先安装 Ollama 并确保其位于 PATH 中。", file=sys.stderr)
        return 1

    results = []
    if not args.skip_embed:
        results.append(pull(args.embed))
    if not args.skip_llm:
        results.append(pull(args.llm))

    print("\n" + "=" * 56)
    print("模型准备完成" if results and all(results) else "部分模型拉取失败，请检查网络后重试。")
    print("=" * 56)
    return 0 if results and all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
