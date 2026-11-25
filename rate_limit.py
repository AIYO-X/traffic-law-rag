# -*- coding: utf-8 -*-
"""
基础限流（令牌桶）与访问日志
============================
为问答接口提供轻量防护：按客户端键（会话 ID 或 IP）做令牌桶限流，
并对每次请求记录访问日志。纯内存 + 标准库实现，无外部依赖。

令牌桶原理：
- 每个键维护一个桶，容量 capacity，以 rate/秒 的速度补充令牌；
- 每次请求消耗 1 个令牌；令牌不足则拒绝（429）；
- 用「惰性补充」：不启动后台线程，而是在每次请求时按时间差计算应补充的令牌，
  既精确又零开销。

线程安全：内部用 threading.Lock 保护桶状态。
"""

import sys
import threading
import time
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402


class TokenBucketLimiter:
    """令牌桶限流器。"""

    def __init__(self, rate: float = None, capacity: int = None):
        self.rate = rate if rate is not None else Config.RATE_LIMIT_PER_SECOND
        self.capacity = capacity if capacity is not None else Config.RATE_LIMIT_CAPACITY
        self._buckets: Dict[str, dict] = {}  # key -> {"tokens": float, "updated": float}
        self._lock = threading.Lock()

    def _refill(self, bucket: dict) -> None:
        """按时间差惰性补充令牌。"""
        now = time.time()
        elapsed = now - bucket["updated"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.rate)
        bucket["updated"] = now

    def allow(self, key: str) -> bool:
        """尝试消耗 1 个令牌，返回是否放行。"""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = {"tokens": float(self.capacity), "updated": time.time()}
                self._buckets[key] = bucket
            self._refill(bucket)
            if bucket["tokens"] >= 1.0:
                bucket["tokens"] -= 1.0
                return True
            return False

    def reset(self, key: str) -> None:
        """重置某键的桶（如会话清空时）。"""
        with self._lock:
            self._buckets.pop(key, None)

    def cleanup(self, max_idle: float = 3600.0) -> None:
        """清理长时间未活跃的桶，避免内存无限增长。"""
        now = time.time()
        with self._lock:
            stale = [k for k, b in self._buckets.items() if now - b["updated"] > max_idle]
            for k in stale:
                self._buckets.pop(k, None)


class AccessLogger:
    """访问日志：写文件 + 标准输出。"""

    def __init__(self, log_file=None):
        self.log_file = Path(log_file) if log_file else Config.ACCESS_LOG_FILE
        self._lock = threading.Lock()

    def log(self, client_key: str, path: str, status: int, duration_ms: float, note: str = ""):
        """记录一行访问日志。"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts} | {client_key} | {path} | {status} | {duration_ms:.0f}ms"
        if note:
            line += f" | {note}"
        # 标准输出
        print(line)
        # 落盘（尽力而为，失败不影响主流程）
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.log_file.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    limiter = TokenBucketLimiter(rate=1.0, capacity=3)
    print("限流测试（rate=1/s, capacity=3，应放行 3 次后拒绝）：")
    for i in range(5):
        print(f"  第 {i + 1} 次:", "放行" if limiter.allow("demo") else "拒绝")
