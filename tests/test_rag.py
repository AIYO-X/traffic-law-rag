# -*- coding: utf-8 -*-
"""
自动化测试
===========
不依赖 Ollama 的测试可直接运行；依赖本地模型的测试用 pytest.mark.skipif 跳过。

运行：
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from knowledge_base.loader import build_chunks, chunk_stats, chunk_text, load_knowledge_items  # noqa: E402
from rag.prompts import build_messages, build_system_prompt, format_context  # noqa: E402


def test_knowledge_scale():
    """要求不少于 50 条有效知识片段。"""
    items = load_knowledge_items()
    assert len(items) >= 50
    for item in items:
        assert item["id"] and item["title"] and item["content"] and item["source"]


def test_knowledge_covers_new_domains():
    """知识库应覆盖铁路高铁、城市轨道交通、民用航空等新领域。"""
    items = load_knowledge_items()
    categories = {item["category"] for item in items}
    for cat in ["铁路与高铁", "城市轨道交通", "民用航空", "高速公路"]:
        assert cat in categories, f"缺少分类 {cat}"
    # 每个新领域至少 5 条知识
    from collections import Counter
    counter = Counter(item["category"] for item in items)
    for cat in ["铁路与高铁", "城市轨道交通", "民用航空"]:
        assert counter[cat] >= 5, f"{cat} 条目不足"


def test_chunks_keep_source_metadata():
    """分段后必须保留标题/出处/分类，保证答案可溯源。"""
    chunks = build_chunks()
    assert len(chunks) >= 50
    assert len({c["item_id"] for c in chunks}) >= 50
    for chunk in chunks:
        assert chunk["title"] and chunk["source"] and chunk["category"]
        assert chunk["text"].startswith("【")


def test_chunk_stats():
    stats = chunk_stats(build_chunks())
    assert stats["chunk_count"] >= 50
    assert stats["min_chars"] >= Config.MIN_CHUNK_CHARS
    assert len(stats["category_distribution"]) >= 8


def test_long_text_chunking():
    """超长文本应被切分，且相邻块之间存在重叠。"""
    text = "第一句话说明规则。" * 120
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[0][-10:] in chunks[1]


def test_prompt_v3_constraints():
    """v3 Prompt 必须包含角色、引用约束与拒答规则。"""
    results = [{
        "title": "闯红灯的记分与认定方式",
        "source": "《记分管理办法》第十条",
        "content": "违反道路交通信号灯通行，一次记6分。",
    }]
    prompt = build_system_prompt("闯红灯扣几分？", results, version="v3")
    assert "中国交通出行法规智能助手" in prompt
    assert "不得" in prompt
    assert "[1]" in prompt
    assert "闯红灯扣几分？" in prompt


def test_prompt_context_limit():
    results = [{"title": "t", "source": "s", "content": "内容" * 40} for _ in range(5)]
    context = format_context(results, max_chars=200)
    assert len(context) <= 220  # 允许编号/换行带来的少量误差


def test_build_messages_with_history():
    results = [{"title": "t", "source": "s", "content": "c"}]
    messages = build_messages(
        "那罚款呢？", results, history=[{"user": "闯红灯扣几分？", "assistant": "记6分。"}]
    )
    assert messages[0]["role"] == "system"
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == "那罚款呢？"


def test_ngram_dedup():
    from knowledge_base.vector_store import VectorStore

    assert VectorStore._ngram_jaccard("闯红灯记6分", "闯红灯记6分") == 1.0
    assert VectorStore._ngram_jaccard("闯红灯记6分", "醉酒驾驶吊销驾照") < 0.3


def test_flask_routes():
    pytest.importorskip("flask")
    from app import app

    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.post("/api/chat", json={"question": ""}).status_code == 400
    assert client.post("/api/clear", json={}).status_code == 200


def test_modules_importable():
    """占位：确保 core 模块可被导入，未安装 lancedb 时也不应崩溃。"""
    import knowledge_base.vector_store as vs
    import rag.pipeline as pipeline

    assert hasattr(vs, "VectorStore")
    assert hasattr(pipeline, "RAGPipeline")


# ==========================================================
# 话题边界守卫（不依赖 Ollama）
# ==========================================================
def test_guard_off_topic_basic():
    """含交通术语的相关问题放行，明显离题的问题拦截。"""
    from rag.guard import is_off_topic

    for q in ["闯红灯扣几分？", "酒驾和醉驾怎么区分？", "高速错过出口能倒车吗？",
              "斑马线不礼让行人怎么处罚？", "驾驶证被扣满12分怎么办？"]:
        assert is_off_topic(q) is False, q
    for q in ["今天天气怎么样？", "帮我写一首关于春天的诗", "1+1等于几？",
              "怎么做红烧肉？", "推荐几部好看的电影", "股票明天会涨吗？"]:
        assert is_off_topic(q) is True, q


def test_guard_followup_and_topic_switch():
    """多轮追问放行，但话题突变仍然拦截（放行条件需收紧）。"""
    from rag.guard import is_off_topic

    assert is_off_topic("那我该怎么办呢？", has_history=True) is False
    assert is_off_topic("那罚款呢？", has_history=True) is False
    assert is_off_topic("再具体说说", has_history=True) is False
    assert is_off_topic("今天天气怎么样？", has_history=True) is True
    assert is_off_topic("帮我写一首关于春天的诗", has_history=True) is True
    assert is_off_topic("推荐几部好看的电影", has_history=True) is True


def test_guard_new_domains_allowed():
    """高铁、地铁、飞机、无人机等新领域问题应判定为领域内（放行）。"""
    from rag.guard import is_off_topic

    for q in ["充电宝能带上飞机吗？", "高铁上吸烟怎么处罚？", "地铁逃票会被罚款吗？",
              "无人机要实名登记吗？", "动车组儿童票怎么买？", "航班延误能赔偿吗？",
              "高铁能带多少行李？", "地铁首末班车几点？"]:
        assert is_off_topic(q) is False, q


def test_guard_still_blocks_unrelated():
    """新领域扩展后，纯离题问题仍应拦截。"""
    from rag.guard import is_off_topic

    for q in ["今天天气怎么样？", "帮我写一首关于春天的诗", "推荐几部电影"]:
        assert is_off_topic(q) is True, q


def test_guard_reply_is_polite():
    """引导语必须委婉、说明边界并主动把话题拉回交通法规。"""
    from rag.guard import pick_reply

    for query in ["今天天气怎么样？", "帮我写一首诗"]:
        reply = pick_reply(query)
        assert len(reply) > 40
        assert "交通出行法规" in reply
        assert any(word in reply for word in ["不好意思", "帮不上", "不如", "呢", "啦"])
        assert not any(word in reply for word in ["禁止提问", "无效问题", "错误"])


# ==========================================================
# RAG 流程：离题短路 + 中途打断（用桩，不依赖 Ollama）
# ==========================================================
class _FakeStore:
    embedder = None

    def is_ready(self):
        return True

    def count(self):
        return 65

    def search_by_text(self, query, top_k=None):
        return [{
            "chunk_id": "c1", "item_id": "i", "title": "恶劣天气与车辆侧滑的处置",
            "source": "《实施条例》第四十六条", "tags": [], "content": "恶劣天气应减速慢行",
            "category": "特殊情形", "score": 0.677,
            "semantic_score": 0.677, "keyword_boost": 0.18,
        }]


class _FakeLLM:
    def __init__(self):
        self.called = False

    def chat_stream(self, messages, should_stop=None, max_tokens=None):
        self.called = True
        self.max_tokens = max_tokens
        yield "闯红灯一次记"
        yield "6分 [1]。"


def test_pipeline_off_topic_skips_llm(monkeypatch):
    """离题问题不得调用大模型，且 done 事件带 off_topic 标记。"""
    from rag.pipeline import RAGPipeline

    monkeypatch.setattr(Config, "GUARD_ENABLED", True)
    llm = _FakeLLM()
    events = list(RAGPipeline(store=_FakeStore(), llm=llm).answer_stream("今天天气怎么样？"))
    assert llm.called is False
    assert events[-1]["event"] == "done"
    assert events[-1]["data"]["off_topic"] is True
    assert events[-1]["data"]["sources"] == []
    assert "交通出行法规" in events[-1]["data"]["answer"]


def test_pipeline_stop_mid_stream():
    """中途打断应 yield stopped，不发 done，且不写入历史。"""
    from rag.pipeline import RAGPipeline

    flag = {"stop": False}

    class StopLLM(_FakeLLM):
        def chat_stream(self, messages, should_stop=None, max_tokens=None):
            self.called = True
            yield "已经生成的一半"
            flag["stop"] = True
            yield "不应出现的后半"

    events = list(
        RAGPipeline(store=_FakeStore(), llm=StopLLM()).answer_stream(
            "闯红灯扣几分？", should_stop=lambda: flag["stop"]
        )
    )
    names = [e["event"] for e in events]
    assert "stopped" in names and "done" not in names
    assert all("不应出现" not in str(e.get("data", "")) for e in events)


def test_pipeline_query_rewrite():
    """多轮追问应做指代消解，把历史主题词融入检索查询。"""
    from rag.pipeline import RAGPipeline

    p = RAGPipeline(store=_FakeStore(), llm=_FakeLLM())
    # 无历史：原样返回
    assert p._rewrite_query("闯红灯扣几分？", None) == "闯红灯扣几分？"
    # 指代性追问 + 历史含主题词：应融合历史主题（「罚款」虽命中词表但仍需指代消解）
    rewritten = p._rewrite_query("那罚款呢？", [{"user": "闯红灯扣几分？", "assistant": "记6分"}])
    assert "闯红灯" in rewritten and "那罚款呢" in rewritten
    # 追问但历史无交通术语：原样返回
    assert p._rewrite_query("那怎么办？", [{"user": "你好", "assistant": "你好"}]) == "那怎么办？"
    # 非追问（不含 FOLLOWUP_MARKERS）且无术语：不做融合（避免误伤）
    assert p._rewrite_query("你好", [{"user": "闯红灯扣几分？", "assistant": "记6分"}]) == "你好"


def test_flask_stop_route():
    """/api/stop 应始终返回 200，且不会因无进行中的生成而报错。"""
    pytest.importorskip("flask")
    from app import app

    client = app.test_client()
    resp = client.post("/api/stop", json={"session_id": "unit-test"})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


# ==========================================================
# 回答详细程度自适应（不依赖 Ollama）
# ==========================================================
def test_verbosity_detect_concise():
    """直接数值/是非查询 → concise。"""
    from rag.verbosity import detect_verbosity

    for q in ["闯红灯扣几分？", "高速超速20%罚多少钱？", "临时停车能停多久？"]:
        assert detect_verbosity(q) == "concise", q


def test_verbosity_detect_detailed():
    """展开性/情形分析/流程步骤 → detailed。"""
    from rag.verbosity import detect_verbosity

    for q in ["酒驾和醉驾怎么区分？", "我追尾了对方逃逸了，我该怎么办？",
              "驾驶证记分周期是怎么算的？", "变道剐蹭谁负责？"]:
        assert detect_verbosity(q) == "detailed", q


def test_verbosity_detect_standard():
    """一般法规问答 → standard。"""
    from rag.verbosity import detect_verbosity

    assert detect_verbosity("斑马线不礼让行人会被处罚吗？") == "standard"


def test_verbosity_max_tokens_monotonic():
    """详细程度越高，生成长度上限越大。"""
    from rag.verbosity import verbosity_max_tokens

    assert (
        verbosity_max_tokens("concise")
        < verbosity_max_tokens("standard")
        < verbosity_max_tokens("detailed")
    )


def test_pipeline_passes_verbosity_to_llm():
    """简单问题应给模型传入更小的 max_tokens，复杂问题应更大。"""
    from rag.pipeline import RAGPipeline

    llm = _FakeLLM()
    p = RAGPipeline(store=_FakeStore(), llm=llm)

    list(p.answer_stream("闯红灯扣几分？"))
    concise_tokens = llm.max_tokens

    list(p.answer_stream("我追尾了对方逃逸了，我该怎么办？"))
    detailed_tokens = llm.max_tokens

    assert concise_tokens < detailed_tokens


def test_prompt_contains_verbosity_instruction():
    """v3 Prompt 应包含「回答详细程度」指令块。"""
    from rag.prompts import build_system_prompt

    results = [{"title": "t", "source": "s", "content": "内容"}]
    prompt = build_system_prompt("闯红灯扣几分？", results, version="v3", verbosity="concise")
    assert "回答详细程度" in prompt


# ---- 以下测试需要本机运行 Ollama 且已构建向量库 ----
try:
    import urllib.request

    urllib.request.urlopen(f"{Config.OLLAMA_HOST}/api/tags", timeout=2)
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="本机未运行 Ollama，跳过端到端测试")
def test_end_to_end_if_ollama_ready():
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    if not pipeline.store.is_ready():
        pytest.skip("向量库尚未构建")
    result = pipeline.answer_sync("闯红灯扣几分？")
    assert result["answer"]
    assert isinstance(result["sources"], list)


# ==========================================================
# 二次精排（rerank）
# ==========================================================
def test_title_hit_rate():
    """标题命中率：问题与标题强命中时返回高值，无命中时返回 0。"""
    from knowledge_base.vector_store import VectorStore

    row = {"title": "闯红灯的记分与认定方式"}
    assert VectorStore._title_hit_rate("闯红灯扣几分？", row) > 0.0
    assert VectorStore._title_hit_rate("今天天气怎么样？", row) == 0.0
    assert VectorStore._title_hit_rate("", row) == 0.0


def test_rerank_sorts_by_final_score():
    """精排应综合语义分、关键词命中与标题命中重新排序。"""
    from knowledge_base.vector_store import VectorStore

    store = VectorStore()
    # 语义分低但标题强命中的条目，精排后应被提到语义分高但无标题命中的条目之前
    candidates = [
        {
            "title": "锂电池与充电宝携带规定",
            "semantic_score": 0.55,
            "keyword_boost": 0.28,
        },
        {
            "title": "校车停靠时的避让义务",
            "semantic_score": 0.706,
            "keyword_boost": 0.0,
        },
    ]
    reranked = store._rerank(candidates, "充电宝能带上飞机吗？")
    assert reranked[0]["title"] == "锂电池与充电宝携带规定"
    assert "title_hit" in reranked[0]
    assert reranked[0]["score"] >= reranked[1]["score"]


# ==========================================================
# 会话存储（JSON 持久化）
# ==========================================================
def test_session_store_persist_and_clear():
    """会话应能持久化、重载、追加与清空。"""
    from session_store import SessionStore

    store = SessionStore(file_path=Config.SESSION_FILE.with_suffix(".test.json"),
                         ttl=3600, max_sessions=10)
    try:
        store.append("sid-test", "闯红灯扣几分？", "记6分")
        store.append("sid-test", "那罚款呢？", "20-200元")
        assert store.get_history("sid-test")[-1]["user"] == "那罚款呢？"

        # 新实例重载，验证持久化
        store2 = SessionStore(file_path=Config.SESSION_FILE.with_suffix(".test.json"),
                              ttl=3600, max_sessions=10)
        assert store2.get_history("sid-test")[-1]["assistant"] == "20-200元"
        store2.clear("sid-test")
        assert store2.get_history("sid-test") == []
    finally:
        import os
        try:
            os.remove(Config.SESSION_FILE.with_suffix(".test.json"))
        except OSError:
            pass


def test_session_store_ttl_expiry():
    """ttl=0 时应立即过期，历史返回空。"""
    from session_store import SessionStore

    store = SessionStore(file_path=Config.SESSION_FILE.with_suffix(".ttl.json"),
                         ttl=0, max_sessions=10)
    try:
        store.append("sid-ttl", "q", "a")
        assert store.get_history("sid-ttl") == []
    finally:
        import os
        try:
            os.remove(Config.SESSION_FILE.with_suffix(".ttl.json"))
        except OSError:
            pass


# ==========================================================
# 限流与接口防护
# ==========================================================
def test_token_bucket_limiter():
    """令牌桶：容量耗尽后应拒绝，补充后恢复。"""
    from rate_limit import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate=0.0, capacity=3)  # rate=0 不补充，便于确定性测试
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False  # 容量耗尽
    limiter.reset("k")
    assert limiter.allow("k") is True   # 重置后恢复


def test_flask_question_length_limit():
    """超长问题应返回 400。"""
    pytest.importorskip("flask")
    from app import app

    client = app.test_client()
    resp = client.post("/api/chat", json={"question": "长" * (Config.MAX_QUESTION_CHARS + 1)})
    assert resp.status_code == 400
    assert "过长" in resp.get_json()["error"]

