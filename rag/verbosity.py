# -*- coding: utf-8 -*-
"""
回答详细程度自适应（Answer Verbosity）
=====================================
根据用户问题的具体类型与措辞，判定回答应有多详细，从而在 Prompt 中注入
对应的篇幅指令，并动态调整生成长度上限（num_predict）。

为什么需要它：
--------------------------------------------------
统一的「先结论后依据」模板对简单查询和复杂情形一视同仁，会导致两种
体验问题：
1. 「闯红灯扣几分？」这种一句话查询，如果模型照模板长篇展开，既慢又啰嗦；
2. 「我追尾了，对方跑了，我该怎么办？」这种复杂情形，如果回答被固定
   的 token 上限截断，关键处理步骤说不完整。
因此按问题特征分三档，让模型「问得简单就答得简短，问得复杂就答得充分」。

三档说明：
    concise  —— 直接事实查询（扣几分 / 罚多少 / 限速多少 / 要不要 / 会不会）
                只给结论 + 简短依据，不展开。
    standard —— 一般法规问答（默认档），先结论、列依据、适度说明。
    detailed —— 开放 / 情形分析 / 流程步骤类（为什么 / 怎么办 / 怎么处理 /
                如何 / 区别 / 流程 / 步骤 / 我…会…吗），需完整展开依据、
                处理建议与注意事项。

判定优先级：
    1. 命中「展开信号词」→ detailed
    2. 命中「简洁信号词」→ concise
    3. 问题较长（可能描述了具体情形）→ detailed
    4. 其余 → standard
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ------------------------------------------------------------------
# 展开信号：出现这些词说明用户要的是分析、步骤、区别或完整处理建议
# ------------------------------------------------------------------
DETAIL_MARKERS: tuple = (
    "为什么", "怎么办", "怎么处理", "如何处理", "怎么判定", "如何认定",
    "怎么算", "如何计算", "怎么计算", "如何", "怎样", "哪些",
    "什么情况", "什么情形", "区别", "区分",
    "流程", "步骤", "程序", "责任", "谁负责", "谁担责", "怎么赔偿",
    "怎么理赔", "注意什么", "需要什么", "要什么手续", "怎么复议",
    "怎么申诉", "怎么维权", "具体", "详细", "完整", "全面", "举例",
    "会吗", "吗，", "会怎样", "后果", "怎么判", "认定标准",
)

# ------------------------------------------------------------------
# 简洁信号：出现这些词说明用户只要一个明确数值/是非结论
# ------------------------------------------------------------------
CONCISE_MARKERS: tuple = (
    "扣几分", "记几分", "几分", "罚多少", "罚多少钱", "罚款多少",
    "多少元", "限速多少", "多少钱", "要不要", "需不需要", "用不用",
    "能不能", "可以吗", "行吗", "对吗", "是不是", "对不对",
    "几公里", "几米", "几点", "几点起", "多少岁", "多久", "多长时间",
)

# 简短展开词的阈值：问题长度超过该值且没有明确简洁信号时，视为描述具体情形
DETAIL_LEN_THRESHOLD = 24


def detect_verbosity(query: str) -> str:
    """
    判定回答详细程度，返回 'concise' | 'standard' | 'detailed'。

    参数
    ----
    query : 用户原始问题
    """
    q = (query or "").strip()
    if not q:
        return "standard"

    # 1. 展开信号优先（复杂情形分析 > 简洁查询）
    if any(m in q for m in DETAIL_MARKERS):
        return "detailed"

    # 2. 简洁信号 → 直接事实查询
    if any(m in q for m in CONCISE_MARKERS):
        return "concise"

    # 3. 问题较长，可能描述具体情形，需要充分展开
    if len(q) >= DETAIL_LEN_THRESHOLD:
        return "detailed"

    # 4. 默认档
    return "standard"


# 各档对应的生成长度上限（num_predict，token 数）。
# 法规回答通常简洁，但复杂情形需要更多空间，这里给三档分别设限，
# 既避免简单问题冗余拖慢，又防止复杂问题被截断。
VERBOSITY_MAX_TOKENS = {
    "concise": 160,
    "standard": 384,
    "detailed": 640,
}

# 各档注入 Prompt 的篇幅指令
VERBOSITY_INSTRUCTIONS = {
    "concise": (
        "这是一个简单的事实查询，请直接、简洁地给出结论和依据，"
        "控制在 1~2 句话以内，不要展开背景、步骤或注意事项。"
    ),
    "standard": (
        "请先给出结论，再列出关键依据，条理清晰即可，"
        "不要过度展开无关内容。"
    ),
    "detailed": (
        "这是一个需要展开回答的问题，请完整地："
        "先给出明确结论，再逐条列出适用的法规依据，"
        "接着给出具体的处理建议或操作步骤，最后提醒注意事项"
        "（如以交警现场认定为准、及时报警或保存证据等）。"
        "可以适当分段、分点，确保用户能据此采取行动。"
    ),
}


def verbosity_instruction(level: str) -> str:
    """返回某档的篇幅指令文本，未知档位回退到 standard。"""
    return VERBOSITY_INSTRUCTIONS.get(level, VERBOSITY_INSTRUCTIONS["standard"])


def verbosity_max_tokens(level: str) -> int:
    """返回某档的生成长度上限，未知档位回退到 standard。"""
    return VERBOSITY_MAX_TOKENS.get(level, VERBOSITY_MAX_TOKENS["standard"])


if __name__ == "__main__":
    samples = [
        ("闯红灯扣几分？", "concise"),
        ("高速超速20%罚多少钱？", "concise"),
        ("酒驾和醉驾怎么区分？", "detailed"),
        ("我追尾了对方逃逸了，我该怎么办？", "detailed"),
        ("斑马线不礼让行人会被处罚吗？", "standard"),
        ("变道剐蹭谁负责？", "detailed"),
        ("临时停车能停多久？", "concise"),
        ("驾驶证记分周期是怎么算的？", "detailed"),
    ]
    for q, expect in samples:
        got = detect_verbosity(q)
        flag = "OK " if got == expect else "?? "
        print(f"{flag}{q:<22} 判定={got:<10} 期望={expect}")
