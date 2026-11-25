# -*- coding: utf-8 -*-
"""
会话存储模块（JSON 文件持久化）
================================
用本地 JSON 文件替代进程内存字典保存会话历史，使服务重启后对话上下文
不丢失；同时提供过期清理与最久未活跃淘汰，避免文件无限膨胀。

设计要点：
1. 内存中维护一份会话字典作为「热缓存」，读写先走内存，再异步/同步落盘；
2. 落盘采用「读-改-写」全量序列化 + 文件锁（threading.Lock），单机多线程下
   保证一致；多进程场景建议改用 Redis/数据库（见 README）；
3. 每次读写记录 last_active 时间戳，读取历史时顺带清理过期会话；
4. 数据文件损坏时安全降级：重命名为 .bak 并从头开始，不让异常阻断服务。

数据格式：
    {
        "session_id": {
            "turns": [{"user": "...", "assistant": "..."}, ...],
            "last_active": 1730000000.0
        },
        ...
    }
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402


class SessionStore:
    """会话历史的 JSON 文件持久化存储。"""

    def __init__(
        self,
        file_path=None,
        ttl: int = None,
        max_sessions: int = None,
    ):
        self.file_path = Path(file_path) if file_path else Path(Config.SESSION_FILE)
        self.ttl = ttl if ttl is not None else Config.SESSION_TTL
        self.max_sessions = max_sessions if max_sessions is not None else Config.MAX_SESSIONS
        self._lock = threading.Lock()
        self._sessions: Dict[str, dict] = {}
        self._load()

    # ---------- 文件读写 ----------
    def _load(self) -> None:
        """从磁盘加载会话；文件不存在或损坏时安全降级为空。"""
        if not self.file_path.exists():
            return
        try:
            with self.file_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._sessions = data
        except (json.JSONDecodeError, OSError, ValueError):
            # 损坏时备份并重置，不让异常阻断服务启动
            try:
                backup = self.file_path.with_suffix(".json.bak")
                os.replace(self.file_path, backup)
            except OSError:
                pass
            self._sessions = {}

    def _save(self) -> None:
        """全量落盘（调用方需已持有 _lock）。"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # 先写临时文件再原子替换，避免写一半崩溃导致 JSON 损坏
        tmp = self.file_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._sessions, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, self.file_path)

    # ---------- 清理 ----------
    def _evict_if_needed(self) -> None:
        """清理过期会话，并在超过上限时淘汰最久未活跃的会话。"""
        now = time.time()
        # 1. 过期清理
        expired = [
            sid for sid, s in self._sessions.items()
            if now - float(s.get("last_active", now)) > self.ttl
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

        # 2. 超量淘汰（按最久未活跃）
        overflow = 0
        if len(self._sessions) > self.max_sessions:
            ordered = sorted(
                self._sessions.items(),
                key=lambda kv: float(kv[1].get("last_active", 0)),
            )
            overflow = len(self._sessions) - self.max_sessions
            for sid, _ in ordered[:overflow]:
                self._sessions.pop(sid, None)

        if expired or overflow > 0:
            self._save()

    # ---------- 对外接口 ----------
    def get_history(self, session_id: str, rounds: int = None) -> List[Dict[str, str]]:
        """返回某会话最近 rounds 轮问答历史。"""
        rounds = rounds or Config.HISTORY_ROUNDS
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            turns = session.get("turns", [])
            # 读取即活跃，更新 last_active（顺带触发过期清理）
            session["last_active"] = time.time()
            self._evict_if_needed()
            return list(turns[-rounds:])

    def append(self, session_id: str, question: str, answer: str) -> None:
        """追加一轮问答并持久化。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = {"turns": [], "last_active": time.time()}
                self._sessions[session_id] = session
            session["turns"].append({"user": question, "assistant": answer})
            session["turns"] = session["turns"][-Config.HISTORY_ROUNDS:]
            session["last_active"] = time.time()
            self._evict_if_needed()
            self._save()

    def clear(self, session_id: str) -> None:
        """清空某会话并持久化。"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions.pop(session_id, None)
                self._save()

    def count(self) -> int:
        """当前会话总数（不含过期清理副作用之外的读取）。"""
        with self._lock:
            return len(self._sessions)


if __name__ == "__main__":
    store = SessionStore()
    print(f"会话存储文件：{store.file_path}")
    print(f"当前会话数：{store.count()}")
