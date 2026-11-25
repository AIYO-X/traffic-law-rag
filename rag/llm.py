# -*- coding: utf-8 -*-
"""
Ollama 生成模型封装
===================
通过 POST /api/chat 调用本地大模型，支持流式（逐 token）与非流式两种模式。
流式响应用于网页端打字机效果；非流式用于同步接口与自动化测试。

可中断：chat_stream 接受 should_stop 回调，置位后立即关闭与 Ollama 的
流式连接并退出，使后端推理真正停止（CPU 推理场景下能明显节省算力）。
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402

from config import Config  # noqa: E402
from rag.embedder import OllamaError  # noqa: E402


class OllamaLLM:
    """封装 Ollama 文本生成接口。"""

    def __init__(self, model: str = None, host: str = None, timeout: int = None):
        self.model = model or Config.LLM_MODEL
        self.host = (host or Config.OLLAMA_HOST).rstrip("/")
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self._session = requests.Session()

    def _options(self, max_tokens: Optional[int] = None) -> Dict:
        options = {
            "temperature": Config.TEMPERATURE,
            "top_p": Config.TOP_P,
            "num_predict": max_tokens if max_tokens is not None else Config.MAX_TOKENS,
            "num_ctx": Config.NUM_CTX,
            "num_gpu": Config.NUM_GPU,
        }
        return options

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        should_stop: Optional[Callable[[], bool]] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        流式生成：逐个 yield 文本片段，供 SSE 实时推送。

        should_stop : 无参回调，返回 True 时立即中断生成并关闭 Ollama 连接。
        max_tokens   : 覆盖本次生成的 token 上限（None 时用全局 Config.MAX_TOKENS）。
        """
        if should_stop and should_stop():
            return

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self._options(max_tokens),
            "keep_alive": Config.KEEP_ALIVE,
        }
        try:
            with self._session.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=(10, self.timeout),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    # 每次收到数据前检查取消标记，及时断开推理连接
                    if should_stop and should_stop():
                        response.close()
                        return
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 忽略心跳或非 JSON 行
                    if chunk.get("error"):
                        yield f"\n\n【模型返回错误】{chunk['error']}"
                        return
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except requests.ConnectionError:
            yield "\n\n【连接失败】无法连接 Ollama 服务，请先运行 `ollama serve`。"
        except requests.Timeout:
            yield "\n\n【响应超时】模型推理时间过长，请稍后重试或换用更小的模型。"
        except requests.RequestException as exc:
            yield f"\n\n【请求失败】{exc}"

    def chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        """非流式生成，返回完整回答（便于测试与同步接口）。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(max_tokens),
            "keep_alive": Config.KEEP_ALIVE,
        }
        try:
            response = self._session.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise OllamaError(f"模型返回错误：{data['error']}")
            return data.get("message", {}).get("content", "")
        except requests.ConnectionError as exc:
            raise OllamaError(
                f"无法连接 Ollama（{self.host}），请先运行 `ollama serve`。"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaError(f"模型响应超时（{self.timeout} 秒）。") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"调用生成模型失败：{exc}") from exc

    def check_health(self) -> tuple:
        """检查服务连通性与生成模型是否已下载。"""
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
                f"未找到生成模型 {self.model}（已安装：{names or '无'}）。"
                f"请先运行 `ollama pull {self.model}`。"
            )
        return True, f"Ollama 服务正常，生成模型 {self.model} 已就绪。"


if __name__ == "__main__":
    llm = OllamaLLM()
    ok, message = llm.check_health()
    print(message)
    if ok:
        for piece in llm.chat_stream(
            [{"role": "user", "content": "用一句话说明什么是闯红灯。"}]
        ):
            print(piece, end="", flush=True)
        print()
