from repoagent.context.manager import group_turns, estimate_messages
from repoagent.loop import build_context
from repoagent.state import Message, ToolCall


def test_parallel_tool_calls_are_compressed():
    """一步内 20 个并行工具调用，是一组 21 条 —— 压缩不能因为组数少就跳过。"""
    msgs = [Message(role="user", content="读 20 个文件")]
    msgs.append(Message(role="assistant", content="", tool_calls=[
        ToolCall(id=f"c{i}", name="read_file", args={"path": f"f{i}.txt"}) for i in range(20)
    ]))
    for i in range(20):
        msgs.append(Message(role="tool", content="填充内容。" * 800, tool_call_id=f"c{i}"))

    assert len(group_turns(msgs)) == 2          # 确认这就是 1 个大组
    out = build_context(msgs)
    assert estimate_messages(out) < 20_000, "并行工具调用也必须被进预算"