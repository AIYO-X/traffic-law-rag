# -*- coding: utf-8 -*-
"""
Ollama 嵌入模型封装
===================
通过 Ollama HTTP API 调用 nomic-embed-text 等嵌入模型，把文本转为向量。

接口兼容处理：
- 优先使用新版批量接口 POST /api/embed（支持一次传多条文本，效率高）；
- 若服务版本较旧不支持，则回退到 POST /api/embeddings 逐条调用。

错误处理的设计目标：让人一眼看出「是服务没启动」还是「模型没下载」。
"""

import sys
from pathlib import Path
from typing import List, Union

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

from config import Config  # noqa: E402


class OllamaError(RuntimeError):
    """Ollama 服务不可用、模型缺失或接口返回异常。"""


class OllamaEmbedder:
    """封装 Ollama 嵌入接口。"""

    def __init__(self, model: str = None, host: str = None, timeout: int = None):
        self.model = model or Config.EMBED_MODEL
        self.host = (host or Config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self._session = requests.Session()

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """把一条或多条文本转为向量，始终返回二维列表。"""
        items = [texts] if isinstance(texts, str) else list(texts)
        if not items:
            return []
        if any(not str(t).strip() for t in items):
            raise ValueError("嵌入文本不能为空字符串。")
        return self._embed_batch([str(t) for t in items])

    def embed_one(self, text: str) -> List[float]:
        """单条文本嵌入的便捷方法。"""
        return self.embed([text])[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        # 新版批量接口：一次请求完成所有文本的嵌入
        try:
            response = self._session.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
                timeout=self.timeout,
            )
            if response.ok:
                embeddings = response.json().get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return [[float(v) for v in vec] for vec in embeddings]
        except requests.RequestException:
            pass  # 旧版 Ollama 不支持该接口时回退

        # 旧版接口：逐条调用
        vectors: List[List[float]] = []
        for text in texts:
            try:
                response = self._session.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                vectors.append([float(v) for v in response.json()["embedding"]])
            except requests.ConnectionError as exc:
                raise OllamaError(
                    f"无法连接 Ollama 服务（{self.host}）。请先运行 `ollama serve`。"
                ) from exc
            except requests.Timeout as exc:
                raise OllamaError(
                    f"嵌入请求超时（{self.timeout} 秒）。模型首次加载较慢，请稍后重试或增大超时时间。"
                ) from exc
            except (requests.RequestException, KeyError, ValueError) as exc:
                raise OllamaError(
                    f"调用嵌入模型 {self.model} 失败：{exc}。请确认模型已用 "
                    f"`ollama pull {self.model}` 下载。"
                ) from exc
        return vectors

    def check_health(self) -> tuple:
        """检查服务连通性与模型是否已下载，返回 (是否可用, 说明)。"""
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=8)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except requests.ConnectionError:
            return False, f"无法连接 Ollama（{self.host}），请先运行 `ollama serve`。"
        except requests.RequestException as exc:
            return False, f"Ollama 服务响应异常：{exc}"

        base_name = self.model.split(":")[0]
        if not any(name.split(":")[0] == base_name for name in names):
            return False, (
                f"未找到嵌入模型 {self.model}（已安装：{names or '无'}）。"
                f"请先运行 `ollama pull {self.model}`。"
            )
        return True, f"Ollama 服务正常，嵌入模型 {self.model} 已就绪。"

    @property
    def dimension(self) -> int:
        """返回向量维度（会触发一次真实调用，仅用于调试）。"""
        return len(self.embed_one("维度检测"))


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    ok, message = embedder.check_health()
    print(message)
    if ok:
        vector = embedder.embed_one("闯红灯扣几分？")
        print(f"向量维度：{len(vector)}，前 5 个分量：{[round(v, 4) for v in vector[:5]]}")
