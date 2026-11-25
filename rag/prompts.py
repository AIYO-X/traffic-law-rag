# -*- coding: utf-8 -*-
"""
提示词（Prompt）工程模块
========================
提供三个版本的系统 Prompt，便于实验报告中做效果对比：

- v1 朴素版：直接把问题和资料丢给模型，无角色与约束；
- v2 角色版：加入「交通法规助手」角色，要求依据资料回答；
- v3 严谨版（默认）：角色 + 强制引用编号 + 拒答规则 + 具体情形处理指引。

对比结论（详见 README 实验分析）：
v1 容易脱离给出的资料自由发挥，数字与条款可能张冠李戴；
v2 有一定改善，但缺少引用约束，用户无法追溯依据；
v3 通过编号引用和拒答规则显著提升可核查性与一致性，最适合法规问答。
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from rag.verbosity import verbosity_instruction  # noqa: E402

# ---------------- v1：朴素版（对照组） ----------------
PROMPT_V1 = """请回答下面的问题。

【参考资料】
{context}

【问题】
{query}
"""

# ---------------- v2：角色版 ----------------
PROMPT_V2 = """你是一个交通法规助手。请严格根据下面的资料回答问题，不要使用资料以外的信息。

【资料】
{context}

【问题】
{query}
"""

# ---------------- v3：严谨版（推荐） ----------------
PROMPT_V3 = """你是「中国交通出行法规智能助手」，一名严谨、专业的交通安全与出行法规顾问。你的知识范围覆盖道路交通、高速公路、铁路高铁、城市轨道交通（地铁/轻轨）、民用航空等交通出行领域。你的回答关系到用户的出行安全与法律责任，必须准确、克制、可核查。

【回答依据】
你将收到一组从法规知识库检索到的片段。你必须严格基于这些片段回答：
1. 只使用片段中出现的信息，不得凭记忆或常识编造条款号、记分分值、罚款金额、期限、携带品限制或费用标准；
2. 每个关键结论后必须标注来源编号，格式为 [1]、[2]；
3. 仅当所有片段都与问题无关、确实无法作答时，才说「根据现有知识库无法确认该问题」，并建议用户咨询相关主管部门或查阅官方最新规定；若片段中已有相关依据，必须直接作答，不得套用拒答措辞；
4. 若用户描述的是具体情形（例如「我这样会不会被处罚」「我该怎么办」），先给出适用的法规依据，再给出符合法规的处理建议，并提醒最终以主管部门现场认定或官方最新规定为准；
5. 不得提供任何规避执法、伪造证件或逃避责任的建议。

【回答风格】
- 使用简体中文，先给结论，再列依据，条理清晰；
- 涉及数字（记分、罚款、期限、速度、距离、携带品重量/体积）时务必逐字准确；
- 语气温和客观，不恐吓、不武断。

【回答详细程度】
{verbosity}

【检索到的知识片段】
{context}

【用户问题】
{query}
"""

SYSTEM_PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}


def format_context(results: List[Dict[str, Any]], max_chars: Optional[int] = None) -> str:
    """把检索结果格式化为带编号的上下文，并控制总长度。"""
    max_chars = max_chars or Config.MAX_CONTEXT_CHARS
    blocks: List[str] = []
    used = 0
    for index, item in enumerate(results, 1):
        block = (
            f"[{index}] 标题：{item.get('title', '')}\n"
            f"出处：{item.get('source', '')}\n"
            f"内容：{item.get('content', '')}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) if blocks else "（未检索到相关知识片段）"


def build_system_prompt(
    query: str,
    results: List[Dict[str, Any]],
    version: str = "v3",
    max_chars: Optional[int] = None,
    verbosity: str = "standard",
) -> str:
    """构造 system role 提示词。"""
    template = SYSTEM_PROMPTS.get(version)
    if template is None:
        raise ValueError(f"未知的 Prompt 版本：{version}，可选 {list(SYSTEM_PROMPTS)}")
    # 注入回答详细程度指令（v1/v2 无该占位符，需补上避免 format 报错）
    return template.format(
        context=format_context(results, max_chars),
        query=query,
        verbosity=verbosity_instruction(verbosity),
    )


# 历史回答截断的最大字符数：多轮对话里历史 assistant 的完整回答（含引用标注
# 与长篇展开）塞回 Prompt 会引入大量噪声并挤占上下文。这里只保留每条历史回答
# 的前缀（结论通常在最前面），在保证指代可理解的前提下控制上下文膨胀。
_HISTORY_ANSWER_MAX_CHARS = 120


def _trim_history_answer(answer: str) -> str:
    """截断历史回答，只保留结论前缀，去掉引用标注与冗长展开。"""
    text = (answer or "").strip()
    if len(text) <= _HISTORY_ANSWER_MAX_CHARS:
        return text
    # 优先在句末符号处断开，避免截断半个句子
    cut = text.rfind("。", 0, _HISTORY_ANSWER_MAX_CHARS)
    if cut < 0:
        cut = text.rfind("；", 0, _HISTORY_ANSWER_MAX_CHARS)
    if cut < 0:
        cut = _HISTORY_ANSWER_MAX_CHARS
    return text[: cut + 1].strip()


def build_messages(
    query: str,
    results: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
    version: str = "v3",
    verbosity: str = "standard",
) -> List[Dict[str, str]]:
    """构造完整的 Ollama chat messages（system + 历史轮次 + 当前问题）。

    历史处理规则：
    - 只取最近 Config.HISTORY_ROUNDS 轮（默认 5），防止长会话上下文无限膨胀；
    - 历史里的 assistant 回答做截断，只保留结论前缀，减少噪声并省 token。
    """
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": build_system_prompt(query, results, version, verbosity=verbosity),
        }
    ]

    # 只保留最近 N 轮，且跳过无内容的空轮次
    recent = list(history or [])
    if Config.HISTORY_ROUNDS > 0:
        recent = [t for t in recent if t.get("user") or t.get("assistant")][
            -Config.HISTORY_ROUNDS :
        ]

    for turn in recent:
        if turn.get("user"):
            messages.append({"role": "user", "content": str(turn["user"])})
        if turn.get("assistant"):
            messages.append(
                {"role": "assistant", "content": _trim_history_answer(str(turn["assistant"]))}
            )
    messages.append({"role": "user", "content": query})
    return messages


if __name__ == "__main__":
    demo_results = [
        {
            "title": "闯红灯的记分与认定方式",
            "source": "《道路交通安全违法行为记分管理办法》第十条",
            "content": "驾驶机动车违反道路交通信号灯通行的，一次记6分。",
        }
    ]
    print(build_system_prompt("闯红灯扣几分？", demo_results, version="v3"))
