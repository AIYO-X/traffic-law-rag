# -*- coding: utf-8 -*-
"""
RAG 流程编排
============
完整链路：
用户问题 → 向量化 → LanceDB 语义检索 → 话题边界判定 → 构造 System Prompt
        → 本地大模型生成 → 流式返回

对外提供两种调用方式：
- answer_stream()：yield 事件字典，供 Flask 转成 SSE
  （status / sources / token / stopped / done / error）
- answer_sync()：一次性返回完整结果，供测试或同步接口使用

两个增强点：
1. 话题边界守卫：明显离题的问题在检索后、生成前直接返回委婉引导语，
   不调用大模型（详见 rag/guard.py）；
2. 可中断生成：should_stop 回调一路传到 Ollama 流式连接，用户点「停止生成」
   时后端推理会真正停下来，而不只是前端停止显示。
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from knowledge_base.vector_store import VectorStore  # noqa: E402
from rag.guard import hit_domain_terms, is_off_topic, looks_like_followup, pick_reply  # noqa: E402
from rag.llm import OllamaLLM  # noqa: E402
from rag.prompts import build_messages  # noqa: E402
from rag.verbosity import detect_verbosity, verbosity_max_tokens  # noqa: E402


class RAGPipeline:
    """把检索与生成拼接成完整的问答流水线。"""

    def __init__(self, store: VectorStore = None, llm: OllamaLLM = None):
        self.store = store or VectorStore()
        self.llm = llm or OllamaLLM()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """语义检索阶段。"""
        if not self.store.is_ready():
            raise RuntimeError(
                "向量知识库尚未构建。请先运行：python scripts/build_knowledge_base.py"
            )
        return self.store.search_by_text(query, top_k=top_k)

    # 当前问题若不含交通领域术语，且像追问时，才需要依赖历史做指代消解。
    # 若问题本身已含强主题词（如「高铁」），则无需融合，避免历史主题污染检索。
    _GENERIC_REFERENCE_TERMS = ("罚款", "扣分", "记分", "处罚", "怎么办", "这样", "那")

    def _rewrite_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]],
    ) -> str:
        """
        多轮对话的「查询改写（指代消解）」。
        短追问（如「那罚款呢？」「那怎么办」）省略了指代对象，直接用原文检索
        会跑偏。这里把历史最近一轮用户问题中的交通术语补进当前查询，
        让检索能命中上一轮真正讨论的条款。

        升级点：
        1. 只在「当前问题缺强主题术语」时才融合历史，避免「那高铁呢」这类
           已自足的追问被上一轮的「汽车」主题词污染；
        2. 融合时用上一轮问题的「完整原文 + 命中术语」双信号，比只拼术语
           能保留更多语义上下文（如上一轮问的是「高速公路」，追问「限速呢」）。
        """
        if not history:
            return query
        if not looks_like_followup(query):
            return query

        # 当前问题已含明确交通术语（且非「罚款/扣分」等泛化指代词），
        # 说明它已自足，无需融合历史，避免主题被上一轮污染。
        own_terms = [t for t in hit_domain_terms(query) if t not in self._GENERIC_REFERENCE_TERMS]
        if own_terms:
            return query

        # 取最近一轮用户问题，抽取其交通术语作为「指代主题」
        last_user = ""
        for turn in reversed(history):
            if turn.get("user"):
                last_user = str(turn["user"])
                break
        terms = hit_domain_terms(last_user)
        if not terms:
            return query

        # 融合：上一轮原文（截断）+ 命中术语 + 当前追问，保留更完整的语义上下文
        last_user_trimmed = last_user[:80]
        merged = " ".join(terms) + " " + last_user_trimmed + " " + query
        return merged

    def answer_stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        prompt_version: str = "v3",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式问答，yield 结构化事件。"""
        query = (query or "").strip()
        if not query:
            yield {"event": "error", "data": "问题不能为空。"}
            return

        def stopped() -> bool:
            return bool(should_stop and should_stop())

        yield {"event": "status", "data": "正在将问题转换为向量…"}
        # 多轮追问做查询改写（指代消解），让检索命中上一轮讨论的条款
        search_query = self._rewrite_query(query, history)
        try:
            results = self.retrieve(search_query, top_k)
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}
            return

        # ---- 话题边界守卫：离题问题不调用大模型 ----
        if Config.GUARD_ENABLED and is_off_topic(
            query, results, has_history=bool(history)
        ):
            reply = pick_reply(query)
            yield {"event": "sources", "data": []}
            yield {
                "event": "status",
                "data": "这个问题好像不在交通法规范围内，正在温和引导…",
            }
            yield {"event": "token", "data": reply}
            yield {
                "event": "done",
                "data": {
                    "answer": reply,
                    "sources": [],
                    "query": query,
                    "prompt_version": prompt_version,
                    "off_topic": True,
                },
            }
            return

        # 把命中的法规片段透传给前端，体现「检索增强」过程
        yield {"event": "sources", "data": results}

        # 按问题具体类型自适应回答详细程度：简单查询答得简短，复杂情形答得充分
        verbosity = detect_verbosity(query)
        max_tokens = verbosity_max_tokens(verbosity)
        yield {
            "event": "status",
            "data": f"已检索到 {len(results)} 条相关法规片段，正在生成回答…",
        }

        messages = build_messages(
            query, results, history, version=prompt_version, verbosity=verbosity
        )
        answer_parts: List[str] = []
        for token in self.llm.chat_stream(
            messages, should_stop=should_stop, max_tokens=max_tokens
        ):
            if stopped():
                break
            answer_parts.append(token)
            yield {"event": "token", "data": token}

        # 用户主动打断：不写入历史、不发 done，明确回报 stopped
        if stopped():
            yield {"event": "stopped", "data": "已停止生成"}
            return

        answer = "".join(answer_parts).strip()
        yield {
            "event": "done",
            "data": {
                "answer": answer,
                "sources": results,
                "query": query,
                "prompt_version": prompt_version,
                "off_topic": False,
            },
        }

    def answer_sync(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        prompt_version: str = "v3",
    ) -> Dict[str, Any]:
        """同步问答，返回完整结果（用于单测/接口调试）。"""
        final: Optional[Dict[str, Any]] = None
        for event in self.answer_stream(query, history, top_k, prompt_version):
            if event["event"] == "error":
                raise RuntimeError(event["data"])
            if event["event"] == "done":
                final = event["data"]
        if final is None:
            raise RuntimeError("问答流程未正常结束。")
        return final

    def check_health(self) -> Dict[str, Any]:
        """汇总各环节健康状态，供 /api/health 使用。"""
        embed_ok, embed_msg = self.store.embedder.check_health()
        llm_ok, llm_msg = self.llm.check_health()
        return {
            "ollama_embed": {"ok": embed_ok, "message": embed_msg},
            "ollama_llm": {"ok": llm_ok, "message": llm_msg},
            "vector_store": {
                "ok": self.store.is_ready(),
                "message": (
                    f"向量库已就绪，共 {self.store.count()} 条片段"
                    if self.store.is_ready()
                    else "向量库尚未构建"
                ),
            },
            "models": {"embed": Config.EMBED_MODEL, "llm": Config.LLM_MODEL},
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    import json

    print(json.dumps(pipeline.check_health(), ensure_ascii=False, indent=2))
