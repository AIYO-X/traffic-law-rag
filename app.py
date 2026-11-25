# -*- coding: utf-8 -*-
"""
Flask Web 服务
==============
提供网页界面与问答 API：

- GET  /                    问答页面
- GET  /api/health          环境健康检查
- GET  /api/knowledge/stats 知识库统计
- POST /api/chat            SSE 流式问答（推荐）
- POST /api/chat/sync       同步问答（便于测试）
- POST /api/stop            中断当前会话正在进行的生成
- POST /api/clear           清空当前会话上下文

说明：会话历史持久化到本地 JSON 文件（session_store.py），重启不丢失、
带过期清理；生成取消标记保存在进程内存（运行时状态，无需持久化）。
生产环境应改用 Redis / 数据库，并增加会话过期与并发控制。

可中断设计：
    每个会话持有一个 threading.Event。/api/chat 把它作为 should_stop 回调
    一路传入 Ollama 流式连接；用户点「停止生成」时前端中止 SSE 并调用
    /api/stop 置位该 Event，后端在下一次收到 token 前退出并关闭连接，
    使大模型推理真正停止，而不是仅前端停止显示。
"""

import json
import threading
import time
import uuid
from threading import Event
from typing import Dict, List, Optional

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from config import Config
from rag.pipeline import RAGPipeline
from rate_limit import AccessLogger, TokenBucketLimiter
from session_store import SessionStore

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY

_pipeline: Optional[RAGPipeline] = None
_pipeline_lock = threading.Lock()

# 会话历史：JSON 文件持久化，重启不丢失、带过期清理
_sessions = SessionStore()

# 会话 -> 当前生成任务的取消标记（运行时状态，不持久化）
_cancel_events: Dict[str, Event] = {}
_cancel_lock = threading.Lock()

# 接口防护：令牌桶限流 + 访问日志
_limiter = TokenBucketLimiter()
_logger = AccessLogger()


def get_pipeline() -> RAGPipeline:
    """懒加载单例，避免服务启动时就要求 Ollama 可用。"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = RAGPipeline()
    return _pipeline


def get_session_id() -> str:
    """从前端传参或 Cookie 获取会话 ID，没有则新建。"""
    session_id = (
        (request.get_json(silent=True) or {}).get("session_id")
        or request.headers.get("X-Session-ID")
        or request.cookies.get("session_id")
    )
    if not session_id:
        session_id = str(uuid.uuid4())
    return str(session_id)


def get_client_key() -> str:
    """限流与日志使用的客户端标识：优先会话 ID，其次客户端 IP。"""
    return get_session_id() or (request.remote_addr or "unknown")


def check_rate_limit(session_id: str):
    """限流检查：超限返回 (429, jsonify)，未超限返回 None。"""
    if not Config.RATE_LIMIT_ENABLED:
        return None
    if not _limiter.allow(session_id or get_client_key()):
        return jsonify({"ok": False, "error": "请求过于频繁，请稍后再试。"}), 429
    return None


def get_history(session_id: str) -> List[Dict[str, str]]:
    return _sessions.get_history(session_id)


def append_history(session_id: str, question: str, answer: str) -> None:
    """保存本轮问答（持久化到 JSON 文件，带过期清理）。"""
    _sessions.append(session_id, question, answer)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        return jsonify({"ok": True, **get_pipeline().check_health()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503


@app.route("/api/knowledge/stats")
def knowledge_stats():
    try:
        from knowledge_base.loader import build_chunks, chunk_stats, load_knowledge_items

        items = load_knowledge_items()
        chunks = build_chunks(items)
        pipeline = get_pipeline()
        return jsonify(
            {
                "ok": True,
                "knowledge_items": len(items),
                "chunks": chunk_stats(chunks),
                "vector_rows": pipeline.store.count() if pipeline.store.is_ready() else 0,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """SSE 流式问答：status → sources → token... → done。"""
    start = time.time()
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    session_id = get_session_id()
    history = get_history(session_id)

    if not question:
        _logger.log(session_id, "/api/chat", 400, (time.time() - start) * 1000, "空问题")
        return jsonify({"ok": False, "error": "问题不能为空。"}), 400

    # 输入长度上限
    if len(question) > Config.MAX_QUESTION_CHARS:
        _logger.log(session_id, "/api/chat", 400, (time.time() - start) * 1000, "超长输入")
        return jsonify(
            {"ok": False, "error": f"问题过长，请控制在 {Config.MAX_QUESTION_CHARS} 字以内。"}
        ), 400

    # 基础限流
    limited = check_rate_limit(session_id)
    if limited is not None:
        _logger.log(session_id, "/api/chat", 429, (time.time() - start) * 1000, "限流")
        return limited

    try:
        pipeline = get_pipeline()
        # 提前探测向量库，避免流式开始后才报错
        if not pipeline.store.is_ready():
            _logger.log(session_id, "/api/chat", 503, (time.time() - start) * 1000, "向量库未就绪")
            return jsonify(
                {
                    "ok": False,
                    "error": "向量知识库尚未构建，请先运行：python scripts/build_knowledge_base.py",
                }
            ), 503
    except Exception as exc:
        _logger.log(session_id, "/api/chat", 503, (time.time() - start) * 1000, "初始化异常")
        return jsonify({"ok": False, "error": str(exc)}), 503

    _logger.log(session_id, "/api/chat", 200, (time.time() - start) * 1000, "开始流式生成")

    # 为本轮生成创建独立的取消标记（同一会话的新请求会覆盖旧标记）
    cancel = Event()
    with _cancel_lock:
        _cancel_events[session_id] = cancel

    def generate():
        yield ": connected\n\n"
        try:
            for event in pipeline.answer_stream(
                question, history=history, should_stop=cancel.is_set
            ):
                if cancel.is_set():
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                # 离题引导语不计入对话上下文，避免污染后续多轮判定
                if event["event"] == "done" and not event["data"].get("off_topic"):
                    append_history(session_id, question, event["data"]["answer"])
        except Exception as exc:
            # 主动打断时不需要再推送 error（连接通常已被前端关闭）
            if not cancel.is_set():
                error_event = {"event": "error", "data": f"服务内部错误：{exc}"}
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        finally:
            # 只清理「自己」的标记，避免误删同一会话后续新请求创建的标记
            with _cancel_lock:
                if _cancel_events.get(session_id) is cancel:
                    _cancel_events.pop(session_id, None)
        yield 'data: {"event":"close"}\n\n'

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 缓冲，保证实时输出
        },
    )
    response.set_cookie("session_id", session_id, httponly=True, samesite="Lax")
    return response


@app.route("/api/chat/sync", methods=["POST"])
def chat_sync():
    """同步问答接口，便于脚本测试（不用于页面主流程）。"""
    start = time.time()
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    session_id = get_session_id()
    if not question:
        return jsonify({"ok": False, "error": "问题不能为空。"}), 400
    if len(question) > Config.MAX_QUESTION_CHARS:
        return jsonify(
            {"ok": False, "error": f"问题过长，请控制在 {Config.MAX_QUESTION_CHARS} 字以内。"}
        ), 400
    limited = check_rate_limit(session_id)
    if limited is not None:
        return limited
    try:
        result = get_pipeline().answer_sync(question, history=get_history(session_id))
    except Exception as exc:
        _logger.log(session_id, "/api/chat/sync", 500, (time.time() - start) * 1000, "生成异常")
        return jsonify({"ok": False, "error": str(exc)}), 500
    _logger.log(session_id, "/api/chat/sync", 200, (time.time() - start) * 1000, "回答完成")
    # 离题引导语不计入对话上下文
    if not result.get("off_topic"):
        append_history(session_id, question, result["answer"])
    return jsonify({"ok": True, "session_id": session_id, **result})


@app.route("/api/stop", methods=["POST"])
def stop_generation():
    """中断当前会话正在进行的生成（供前端「停止生成」调用）。"""
    session_id = get_session_id()
    with _cancel_lock:
        event = _cancel_events.get(session_id)
    if event is not None:
        event.set()
        return jsonify({"ok": True, "stopped": True, "message": "已请求停止生成。"})
    return jsonify(
        {"ok": True, "stopped": False, "message": "当前没有正在进行的生成。"}
    )


@app.route("/api/clear", methods=["POST"])
def clear_session():
    session_id = get_session_id()
    _sessions.clear(session_id)
    # 清空对话时顺带中断可能仍在进行的生成
    with _cancel_lock:
        event = _cancel_events.pop(session_id, None)
        if event is not None:
            event.set()
    return jsonify({"ok": True, "message": "对话上下文已清空。"})


if __name__ == "__main__":
    Config.ensure_dirs()
    print("=" * 60)
    print("中国交通出行法规 RAG 问答系统已启动")
    print(f"访问地址：http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    print(f"生成模型：{Config.LLM_MODEL} | 嵌入模型：{Config.EMBED_MODEL}")
    print(f"话题守卫：{'开启' if Config.GUARD_ENABLED else '关闭'}")
    print("=" * 60)
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.DEBUG,
        threaded=True,
    )
