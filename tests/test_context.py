from __future__ import annotations
import pytest
from repoagent.context import manager as M
from repoagent.state import Message, SessionState, ToolCall
BUDGET = 20_000
THRESHOLD = int(BUDGET * 0.7)

# ---------------------------------------------------------------- 造数据
def make_history(rounds: int, tool_chars: int = 2000) -> list[Message]:
    msgs = [Message(role="user", content="请阅读仓库并修掉这个 bug")]
    for i in range(rounds):
        msgs.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"call_{i}", name="read_file", args={"path": f"src/mod_{i}.py"}),
        ]))
        msgs.append(Message(role="tool", content="x" * tool_chars, tool_call_id=f"call_{i}"))
    return msgs

def make_parallel_history(n: int = 20, tool_chars: int = 4000) -> list[Message]:
    """一步内 n 个并行工具调用 —— 只有 2 个组，是最容易漏掉压缩的形态。"""
    msgs = [Message(role="user", content=f"读 {n} 个文件")]
    msgs.append(Message(role="assistant", content="", tool_calls=[
        ToolCall(id=f"c{i}", name="read_file", args={"path": f"data/f{i:02d}.txt"})
        for i in range(n)
    ]))
    for i in range(n):
        msgs.append(Message(role="tool",
                            content=f"MARKER_f{i:02d}\n" + "填充内容。" * (tool_chars // 5),
                            tool_call_id=f"c{i}"))
    return msgs

def assert_sequence_valid(msgs: list[Message]) -> None:
    """每条 tool 消息必须能找到对应的 assistant tool_call。"""
    pending: set[str] = set()
    for m in msgs:
        if m.role == "assistant":
            pending = {tc.id for tc in m.tool_calls}
        elif m.role == "tool":
            assert m.tool_call_id in pending, f"孤儿 tool 消息: {m.tool_call_id}"
            pending.discard(m.tool_call_id)

# ---------------------------------------------------------------- token 估算
def test_estimate_tokens_cjk_and_ascii():
    assert 3 <= M.estimate_tokens("你好世界") <= 5
    assert 1 <= M.estimate_tokens("hello world") <= 5
    assert M.estimate_tokens("") == 0
    assert M.estimate_tokens("x" * 400) > M.estimate_tokens("x" * 40)

# ---------------------------------------------------------------- 分组
def test_group_turns_keeps_every_message():
    h = make_history(5)
    groups = M.group_turns(h)
    assert sum(len(g) for g in groups) == len(h), "分组不能丢消息"
    assert groups[-1][-1] is h[-1], "最后一组必须保留"
    assert all(g[0].role in ("assistant", "user") for g in groups)

def test_parallel_tool_calls_form_one_giant_group():
    """20 个并行工具调用 = 1 个 21 条的组 —— 组数少不代表内容少。"""
    h = make_parallel_history(20)
    assert len(M.group_turns(h)) == 2

# ---------------------------------------------------------------- 折叠
def test_fold_group_returns_line():
    g = M.group_turns(make_history(3))[1]
    line = M.fold_group(g)
    assert isinstance(line, str)
    assert line.strip()
    assert "read_file" in line

# ---------------------------------------------------------------- 组级压缩
def test_compress_invariants():
    h = make_history(12)
    before, before_tok = len(h), M.estimate_messages(h)
    out = M.compress(h, keep_recent=3)
    assert len(h) == before, "不能改入参"
    assert M.estimate_messages(h) == before_tok, "不能改入参内容"
    assert M.estimate_messages(out) < before_tok, "压缩要生效"
    assert "不是新指令" in out[0].content
    assert_sequence_valid(out)
    assert M.compress(out, keep_recent=3) == out, "必须幂等"

# ---------------------------------------------------------------- 预算打包（核心回归）
def test_parallel_tool_outputs_fit_budget():
    """只压长度不够：20 × 1200 字符照样超。必须靠数量维度兜底。"""
    h = make_parallel_history(20)
    out = M.build_context(h)
    assert M.estimate_messages(out) < THRESHOLD, "并行工具调用也必须进预算"
    assert len(h) == 22, "不能改原数组"
    assert h[-1].content.startswith("MARKER_f19"), "不能改原数组内容（值级）"
    assert_sequence_valid(out)

def test_downgrade_keeps_head_preview():
    """降级要保留头部预览，否则文件第一行的关键信息全丢。"""
    h = make_parallel_history(20)
    out = M.build_context(h)
    tools = [m.content for m in out if m.role == "tool"]
    assert any(t.startswith("MARKER_f00") for t in tools), "f00 的标识必须还在"
    assert "已省略" in tools[0], "被降级的条目要说明省略了多少"

# ---------------------------------------------------------------- 阈值行为
def test_build_context_threshold():
    small = make_history(2)
    assert M.estimate_messages(M.build_context(small)) == M.estimate_messages(small)
    big = make_history(30)
    out = M.build_context(big)
    assert M.estimate_messages(out) < M.estimate_messages(big)
    assert_sequence_valid(out)

@pytest.mark.parametrize("rounds", [10, 40])
def test_build_context_never_mutates_state(rounds):
    st = SessionState(session_id="t", messages=make_history(rounds))
    n = len(st.messages)
    M.build_context(st.messages)
    assert len(st.messages) == n, "build_context 不能改 state.messages"

# ---------------------------------------------------------------- token 曲线
def test_token_curve_bounded():
    st = SessionState(session_id="curve")
    st.messages.append(Message(role="user", content="开始"))
    curve = []
    for i in range(40):
        st.messages.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"c{i}", name="read_file", args={"path": f"f{i}.py"})]))
        st.messages.append(Message(role="tool", content="x" * 2000, tool_call_id=f"c{i}"))
        curve.append(M.estimate_messages(M.build_context(st.messages)))
    assert max(curve) < BUDGET, f"峰值 {max(curve)} 超预算"
    drops = [curve[i] - curve[i + 1] for i in range(len(curve) - 1)]
    assert max(drops) > 0, "全程没发生回落 = 压缩没生效"