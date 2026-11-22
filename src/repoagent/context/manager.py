import re

from repoagent.state import Message, ToolCall


def estimate_tokens(text: str) -> int:
    """粗估：中日韩字符按 1 token，其余按 4 字符 1 token。"""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3000-\u9fff\uff00-\uffef]", text))
    other = len(text) - cjk
    return cjk + other // 4

def estimate_messages(messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.content)
        for tool_call in message.tool_calls:
            total += estimate_tokens(tool_call.name) + estimate_tokens(str(tool_call.args))

    return total

def group_turns(messages: list[Message]) -> list[list[Message]]:
    groups: list[list[Message]] = []
    cur = []
    for message in messages:
        if message.role == "tool":
            cur.append(message)
            continue
        if cur:
            groups.append(cur)
        cur = [message]
    if cur:
        groups.append(cur)
    return groups

def fold_group(messages: list[Message]) -> str:
    head = messages[0]
    if head.role in ("user", "system"):
        return f"- 用户：{head.content[:80]}"
    if head.role == "assistant" and not head.tool_calls:
        return f"- 助手：{head.content[:80]}"

    parts = []
    for tool_call in head.tool_calls:
        arg = ",".join(f"{k}={str(v)[:40]}" for k, v in tool_call.args.items()) if tool_call.args else ""
        parts.append(f"{tool_call.name}({arg})")

    results =[message for message in messages[1:] if message.role == "tool"]
    status = "/".join(
        "失败" if res.content.startswith(("Error", "FileNotFoundError", "ValueError", "KeyError"))
        else "成功"
        for res in results
    )
    return f"- 调用 {','.join(parts)} -> {status}"

def compress(messages: list[Message], keep_recent: int = 10) -> list[Message]:
    groups = group_turns(messages)
    if len(groups) <= keep_recent:
        return list(messages)

    old, recent = groups[:-keep_recent+1], groups[-keep_recent+1:]
    lines = [fold_group(g) for g in old if g]
    header = (
            "[已折叠的历史操作，仅作背景参考，不是新指令]\n"
            + "\n".join(lines)
    )

    out: list[Message] = [Message(role="user", content=header)]
    for group in recent:
        out.extend(group)

    return out

def shrink_tool_outputs(messages, head=800, tail=400):
    """第一道闸：单条工具输出过长就头尾保留。不删消息，只改内容。"""
    out = []
    for m in messages:
        if m.role == "tool" and len(m.content) > head + tail:
            m2 = m.model_copy(deep=True)
            cut = len(m.content) - head - tail
            m2.content = (
                m.content[:head]
                + f"\n...[已省略 {cut} 字符，如需完整内容请重新 read_file]...\n"
                + m.content[-tail:]
            )
            out.append(m2)
        else:
            out.append(m)
    return out

def _with_content(m, new_content: str):
    m2 = m.model_copy(deep=True)
    m2.content = new_content
    return m2


def pack_tool_outputs(messages, budget_tokens: int = 20_000,
                      ratio: float = 0.45, keep_full: int = 4,
                      head: int = 800, tail: int = 400):
    """按预算打包工具输出。两道闸，缺一不可：

    ① 长度维度：单条结果超长 -> 头尾保留（800/400 是经验值，不是依据）
    ② 数量维度：总量仍超预算 -> 从**最早**的结果开始，降级成一行元信息

    为什么必须有 ②：20 个文件 × 每个压到 1200 字符 = 24000 字符，
    照样超预算。只压长度不控数量，等于没压。
    """
    tool_budget = max(1, int(budget_tokens * ratio))
    out = list(messages)

    # ---------- 闸 1：长度 ----------
    for i, m in enumerate(out):
        if m.role == "tool" and len(m.content) > head + tail:
            cut = len(m.content) - head - tail
            out[i] = _with_content(
                m,
                m.content[:head]
                + f"\n...[已省略 {cut} 字符，需要完整内容请重新调用工具]...\n"
                + m.content[-tail:],
            )

    # ---------- 闸 2：数量 ----------
    tool_idx = [i for i, m in enumerate(out) if m.role == "tool"]
    if not tool_idx:
        return out

    def used() -> int:
        return sum(estimate_tokens(out[i].content) for i in tool_idx)

    # 最近的 keep_full 条永远保留全文；更早的按需降级
    candidates = tool_idx[:-keep_full] if keep_full > 0 else tool_idx
    for i in candidates:
        if used() <= tool_budget:
            break
        orig = len(messages[i].content)
        # 保留头部预览：标识符 / import / 函数签名都在开头，
        # 整条丢成一行元信息等于把唯一线索也删了
        preview = messages[i].content[:300]
        out[i] = _with_content(
            out[i],
            preview
            + f"\n...[已省略 {orig - len(preview)} 字符。"
              f"需要完整内容请重新调用对应工具]",
        )
    return out


def build_context(messages: list[Message], budget_tokens: int = 20_000) -> list[Message]:
    # 先按预算压工具输出（长度 + 数量两个维度）
    msgs = pack_tool_outputs(messages, budget_tokens=budget_tokens)
    if estimate_messages(msgs) <= budget_tokens * 0.7:
        return msgs
    # 仍超预算，再折叠早期轮次
    return compress(msgs)

if __name__ == '__main__':
    # print(estimate_tokens("你好世界"))
    msgs: list[Message] = [Message(role="user", content="看看这个仓库")]
    for i in range(12):
        msgs.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"c{i}", name="read_file", args={"path": f"f{i}.py"}),
        ]))
        msgs.append(Message(role="tool", content="x" * 2000, tool_call_id=f"c{i}"))

    print("前:", estimate_messages(msgs), len(msgs))
    out = compress(msgs)
    print("后:", estimate_messages(out), len(out))

    assert len(msgs) == 25, "原数组不能被改"
    assert out[-1].role == "tool", "最后一组必须保住（Bug 1 的回归测试）"
    assert compress(out) == out, "幂等"