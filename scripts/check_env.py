# -*- coding: utf-8 -*-
"""
环境检测脚本
============
按顺序检查：Python 版本 → 依赖包 → Ollama 服务与模型 → 知识库与向量库。
只依赖 Python 标准库，任何阶段都可独立运行。

用法：python scripts/check_env.py
"""

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402

REQUIRED_PACKAGES = ["flask", "requests", "numpy", "lancedb"]


def line(title, ok, message):
    icon = "[OK]" if ok else "[!!]"
    print(f"  {icon} {title:<22} {message}")


def check_python():
    print("\n[1/4] Python 环境")
    version = sys.version_info
    ok = version.major == 3 and version.minor >= 8
    line("Python 版本", ok, f"{version.major}.{version.minor}.{version.micro}（要求 >= 3.8）")
    return ok


def check_packages():
    print("\n[2/4] 依赖包")
    all_ok = True
    for name in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(name)
        line(name, spec is not None, "已安装" if spec else "缺失，请运行 pip install -r requirements.txt")
        all_ok = all_ok and spec is not None
    return all_ok


def check_ollama():
    print("\n[3/4] Ollama 服务与模型")
    try:
        with urllib.request.urlopen(f"{Config.OLLAMA_HOST}/api/tags", timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        names = [m.get("name", "") for m in data.get("models", [])]
        line("Ollama 服务", True, f"{Config.OLLAMA_HOST} 可访问")
    except Exception as exc:
        line("Ollama 服务", False, f"无法访问（{exc}），请先启动 `ollama serve`")
        return False

    all_ok = True
    for role, model in (("嵌入模型", Config.EMBED_MODEL), ("生成模型", Config.LLM_MODEL)):
        base = model.split(":")[0]
        found = any(name.split(":")[0] == base for name in names)
        line(role, found, f"{model} {'已就绪' if found else '未下载，请运行 ollama pull ' + model}")
        all_ok = all_ok and found
    if names:
        print(f"       已安装模型：{', '.join(names)}")
    return all_ok


def check_data():
    print("\n[4/4] 知识库与向量库")
    path = Config.KNOWLEDGE_FILE
    if not path.exists():
        line("知识库文件", False, f"不存在：{path}")
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        count = len(data.get("items", []))
        line("知识库文件", count >= 50, f"{count} 条知识片段（要求 >= 50）")
        ok = count >= 50
    except Exception as exc:
        line("知识库文件", False, f"解析失败：{exc}")
        return False

    db_dir = Path(Config.LANCEDB_URI)
    ready = db_dir.exists() and any(db_dir.iterdir())
    line("向量库", ready, f"{db_dir} {'已构建' if ready else '未构建，请运行 python scripts/build_knowledge_base.py'}")
    return ok and ready


def main():
    print("=" * 64)
    print("交通法规 RAG 系统 · 环境检测")
    print("=" * 64)
    results = [check_python(), check_packages(), check_ollama(), check_data()]
    print("\n" + "=" * 64)
    if all(results):
        print("全部检查通过！可以启动：python app.py")
    else:
        print("存在未通过项，请按上面的提示逐项处理后再启动。")
        print("提示：Ollama 未安装时不影响代码交付，但无法运行完整问答流程。")
    print("=" * 64)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
