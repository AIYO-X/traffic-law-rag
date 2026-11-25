# -*- coding: utf-8 -*-
"""
全局配置模块
============
所有可调参数集中在此，避免散落在业务代码中。
支持通过环境变量覆盖，便于 dev / prod 切换与容器化部署。

用法：
    from config import Config
    print(Config.LLM_MODEL)
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    # ---------------- Ollama 服务 ----------------
    # 开发环境默认本机 11434；生产环境可指向内网推理服务器
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    # 默认生成模型：qwen2.5:1.5b（约 1GB，CPU 推理约 30+ token/s，速度快）
    # 速度/质量权衡：
    #   qwen2.5:1.5b —— 最快，法规「照着检索片段作答」场景下准确性已够用；
    #   qwen2.5:3b   —— 速度与能力折中，复杂情形理解更好（约 2GB）；
    #   qwen2.5:7b   —— 推理最强，但本机 4GB 显存下只能走 CPU（约 7 token/s，慢）。
    # 通过环境变量 OLLAMA_LLM_MODEL 一键切换，无需改动代码。
    LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:1.5b")
    REQUEST_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # ---------------- 数据与路径 ----------------
    KNOWLEDGE_FILE = BASE_DIR / "data" / "traffic_law_knowledge.json"
    LANCEDB_URI = BASE_DIR / "data" / "lancedb"
    TABLE_NAME = "traffic_law_chunks"

    # ---------------- 文本分段（chunk）----------------
    CHUNK_SIZE = 500          # 单块目标字符数
    CHUNK_OVERLAP = 80        # 相邻块重叠字符数，避免语义被切断
    MIN_CHUNK_CHARS = 30      # 过滤过短片段

    # ---------------- 检索策略 ----------------
    TOP_K = 5                          # 召回片段数量
    SIMILARITY_THRESHOLD = 0.25        # 余弦相似度下限，过滤弱相关
    MAX_CONTEXT_CHARS = 3000           # 注入 Prompt 的上下文上限
    DEDUP_THRESHOLD = 0.92             # 结果去重：内容高度相似时只保留一条
    # 混合检索：语义相似度为主，标题/标签的术语命中作为加权信号。
    # 中文短问句用纯向量检索时区分度不足，加入该权重可显著提升精确条款命中率。
    # 最终得分 = 语义相似度 + KEYWORD_BOOST_WEIGHT × 关键词命中率（0~1）。
    # 实测：0.28 时跨领域问题（如「充电宝带飞机」「无人机登记」）正确条目
    # 语义分过低被漏召回；0.60 时新旧领域 20 组回归均能命中，为经验甜点值。
    KEYWORD_BOOST_WEIGHT = 0.60

    # ---------------- 二次精排（rerank）----------------
    # 检索初排后，对候选做一次轻量二次精排：综合语义分、关键词命中率、
    # 标题精确命中信号，提升跨领域问题（语义分普遍偏低、关键词差异大）的
    # 排序质量。纯代码实现，不引入额外 rerank 模型依赖。
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
    # 二次精排时的额外召回倍数：初排取 top_k*2 条进入精排，精排后再截断回 top_k。
    RERANK_CANDIDATE_FACTOR = 2
    # 标题命中在精排中的加权（标题是最强主题信号，命中时应显著提升排名）
    RERANK_TITLE_BOOST = 0.15

    # ---------------- 话题边界守卫 ----------------    # 离题问题不调用大模型，直接返回委婉引导语（详见 rag/guard.py）。
    # 设为 0 可关闭守卫，便于实验报告中对比「有 / 无领域约束」的效果。
    GUARD_ENABLED = os.getenv("GUARD_ENABLED", "1") == "1"
    # 检索 Top-1 的关键词命中率高于此值时，即使问题未命中术语表也视为领域内。
    # 实测无关问题最高约 0.18，取 0.30 留有安全余量。
    GUARD_MIN_KEYWORD_BOOST = float(os.getenv("GUARD_MIN_KEYWORD_BOOST", "0.30"))

    # ---------------- 生成参数 ----------------
    TEMPERATURE = 0.2         # 法规问答要求稳定、保守
    TOP_P = 0.9
    MAX_TOKENS = 512          # 法规回答通常 100~300 字，上限收敛避免冗余拖慢
    NUM_CTX = 2048            # 上下文窗口，覆盖 system prompt + 检索片段 + 历史即可
    HISTORY_ROUNDS = 5        # 多轮对话保留的历史轮数（避免上下文膨胀）
    # 模型常驻内存时长（秒）。Ollama 默认空闲约 5 分钟就卸载模型，
    # 下次请求需重新加载（数秒~十几秒冷启动）。设为 -1 表示常驻不卸载，
    # 大幅消除连续问答时的冷启动延迟，代价是常驻占用内存。
    KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    # GPU 卸载层数：0 = 纯 CPU 推理；-1 = 全部层卸载到 GPU。
    # 本机 RTX 3050 Ti 仅 4GB 显存，7B 模型（4.7GB）放不下会崩溃；
    # 1.5b/3b 模型可设为 -1 尝试走 GPU（若显存不足 Ollama 会自动回退到 CPU）。
    NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))

    # ---------------- Flask 服务 ----------------
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me-in-prod")

    # ---------------- 会话存储 ----------------
    # 会话历史持久化到本地 JSON 文件，重启服务不丢失；带过期清理。
    # 生产环境可改为 Redis / 数据库（见 README 第九节）。
    SESSION_FILE = BASE_DIR / "data" / "sessions.json"
    # 会话过期时间（秒）：超过该时长未活跃的会话将被清理
    SESSION_TTL = int(os.getenv("SESSION_TTL", str(24 * 3600)))
    # 最大会话数量，超出后按最久未活跃淘汰
    MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))

    # ---------------- 接口防护 ----------------
    # 输入长度上限（字符数），防止超长问题拖垮嵌入/生成
    MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
    # 基础限流：令牌桶。每秒补充的令牌数（rate）与桶容量（capacity）。
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
    RATE_LIMIT_PER_SECOND = float(os.getenv("RATE_LIMIT_PER_SECOND", "1.0"))
    RATE_LIMIT_CAPACITY = int(os.getenv("RATE_LIMIT_CAPACITY", "5"))
    # 访问日志文件（None 表示只输出到 stdout）
    ACCESS_LOG_FILE = BASE_DIR / "data" / "access.log"

    @classmethod
    def ensure_dirs(cls) -> None:
        """确保数据目录存在。"""
        cls.LANCEDB_URI.parent.mkdir(parents=True, exist_ok=True)
        cls.SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls) -> dict:
        """用于环境检测脚本打印配置摘要。"""
        return {
            "Ollama 地址": cls.OLLAMA_HOST,
            "生成模型": cls.LLM_MODEL,
            "嵌入模型": cls.EMBED_MODEL,
            "知识库文件": str(cls.KNOWLEDGE_FILE),
            "向量库目录": str(cls.LANCEDB_URI),
            "Top-K": cls.TOP_K,
            "相似度阈值": cls.SIMILARITY_THRESHOLD,
            "Flask 监听": f"{cls.FLASK_HOST}:{cls.FLASK_PORT}",
        }


# 生产环境提示（README 中展开）：
# 1) 关闭 DEBUG，设置强随机 FLASK_SECRET_KEY；
# 2) OLLAMA_HOST 指向内网地址，不要直接暴露 11434 到公网；
# 3) 使用 gunicorn / waitress 替代 Flask 开发服务器；
# 4) 为 /api/chat 增加限流与会话过期，防止长连接耗尽资源。
