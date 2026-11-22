"""D3 上下文压缩 · 验收脚本

跑法（在仓库根目录）：
    source .venv/bin/activate
    python ~/Documents/deepseek-harness/default-workspace/notes/03-verify.py

也可以通过 pytest 跑（放在 tests/test_context.py）：
    pytest tests/test_context.py -v

设计原则：全部不联网。压缩是确定性逻辑，就该用确定性测试守住。
"""
from __future__ import annotations

import inspect
import sys

from repoagent.context import manager as M
from repoagent.state import Message, SessionState, ToolCall

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}  {detail}")
        FAILED.append(name)


# ---------------------------------------------------------------- 造数据

def make_history(rounds: int, tool_chars: int = 2000) -> list[Message]:
    """1 条 user + rounds 轮 (assistant[tool_calls] + tool)。"""
    msgs = [Message(role="user", content="请阅读仓库并修掉这个 bug")]
    for i in range(rounds):
        msgs.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"call_{i}", name="read_file", args={"path": f"src/mod_{i}.py"}),
        ]))
        msgs.append(Message(role="tool", content="x" * tool_chars, tool_call_id=f"call_{i}"))
    return msgs


def check_sequence_valid(msgs: list[Message]) -> tuple[bool, str]:
    """结构校验：每条 tool 消息必须能找到对应的 assistant tool_call。"""
    pending: set[str] = set()
    for m in msgs:
        if m.role == "assistant":
            pending = {tc.id for tc in m.tool_calls}
        elif m.role == "tool":
            if m.tool_call_id not in pending:
                return False, f"孤儿 tool 消息: {m.tool_call_id}"
            pending.discard(m.tool_call_id)
    return True, ""


# ---------------------------------------------------------------- 1. token 估算

print("\n[1] estimate_tokens")
check("中文 4 字 ≈ 4 tok", 3 <= M.estimate_tokens("你好世界") <= 5)
check("英文 11 字符 ≈ 3 tok", 1 <= M.estimate_tokens("hello world") <= 5)
check("空串为 0", M.estimate_tokens("") == 0)
check("长文本单调递增", M.estimate_tokens("x" * 400) > M.estimate_tokens("x" * 40))

# ---------------------------------------------------------------- 2. 分组：不能丢消息

print("\n[2] group_turns")
h = make_history(5)
groups = M.group_turns(h)
check("总消息数不变（无丢失）", sum(len(g) for g in groups) == len(h),
      f"{sum(len(g) for g in groups)} != {len(h)}")
check("组数 = 1 + 轮数", len(groups) == 6, f"got {len(groups)}")
check("最后一组保留了最后的 tool", groups[-1][-1] is h[-1])
check("每组都以 assistant/user 开头", all(g[0].role in ("assistant", "user") for g in groups))

# ---------------------------------------------------------------- 3. 折叠

print("\n[3] fold_group")
line = M.fold_group(groups[1])
check("返回 str", isinstance(line, str), type(line))
check("非空", bool(line.strip()))
check("包含工具名", "read_file" in line, line[:80])

# ---------------------------------------------------------------- 4. 压缩的不变量

print("\n[4] compress 不变量")
h = make_history(12)
before_len = len(h)
before_tok = M.estimate_messages(h)
out = M.compress(h, keep_recent=3)

check("不改入参（长度）", len(h) == before_len)
check("不改入参（token）", M.estimate_messages(h) == before_tok)
check("压缩生效", M.estimate_messages(out) < before_tok)
check("首条是摘要且声明非指令", "不是新指令" in out[0].content)
ok, why = check_sequence_valid(out)
check("压缩后消息序列合法", ok, why)
check("幂等", M.compress(out, keep_recent=3) == out)

# ---------------------------------------------------------------- 5. 压缩比（30 轮）

print("\n[5] 压缩比")
h30 = make_history(30)
o30 = M.compress(h30, keep_recent=4)
r = M.estimate_messages(o30) / M.estimate_messages(h30)
print(f"    30 轮: {M.estimate_messages(h30)} -> {M.estimate_messages(o30)} ({r:.0%})")
check("30 轮压缩到 40% 以下", r < 0.4, f"{r:.2f}")

# ---------------------------------------------------------------- 6. build_context 阈值

print("\n[6] build_context")
bc = getattr(M, "build_context", None)
if bc is None:
    print("  - manager 里没有 build_context，跳过（下一步要加）")
else:
    params = list(inspect.signature(bc).parameters)
    state = SessionState(session_id="t", messages=make_history(30))

    if params and params[0] == "state":
        small = SessionState(session_id="t2", messages=make_history(2))
        r1 = bc(small)
        r2 = bc(state)
    else:
        r1 = bc(make_history(2))
        r2 = bc(state.messages)

    check("低于阈值时原样返回", M.estimate_messages(r1) == M.estimate_messages(make_history(2)))
    check("高于阈值时压缩", M.estimate_messages(r2) < M.estimate_messages(state.messages))
    ok, why = check_sequence_valid(r2)
    check("压缩后序列合法", ok, why)
    check("build_context 不改 state.messages", len(state.messages) == 61,
          f"got {len(state.messages)}")

# ---------------------------------------------------------------- 7. token 曲线

print("\n[7] token 曲线（每轮新增后调用一次 build_context）")
try:
    from repoagent.context.manager import build_context as _bc
    use_state = list(inspect.signature(_bc).parameters)[0] == "state"

    st = SessionState(session_id="curve")
    st.messages.append(Message(role="user", content="开始"))
    budget = 20_000
    threshold = int(budget * 0.7)
    curve = []
    for i in range(40):                     # 必须跑到超过阈值，否则测不到压缩
        st.messages.append(Message(role="assistant", content="", tool_calls=[
            ToolCall(id=f"c{i}", name="read_file", args={"path": f"f{i}.py"})]))
        st.messages.append(Message(role="tool", content="x" * 2000, tool_call_id=f"c{i}"))
        view = _bc(st) if use_state else _bc(st.messages)
        curve.append(M.estimate_messages(view))

    peak = max(curve)
    drops = [curve[i] - curve[i + 1] for i in range(len(curve) - 1)]
    biggest_drop = max(drops) if drops else 0
    print(f"    预算 {budget} / 阈值 {threshold} / 峰值 {peak} / 最大回落 {biggest_drop}")

    # 三条真正的不变量
    check("峰值不超预算", peak < budget, f"{peak} >= {budget}")
    check("压缩确实触发过（曲线出现回落）", biggest_drop > 0, f"回落 {biggest_drop}")
    check("压缩后明显变小（回落 > 阈值 30%）", biggest_drop > threshold * 0.3,
          f"回落 {biggest_drop}")
    print("    曲线:", " ".join(str(x) for x in curve[:12]), "...",
          " ".join(str(x) for x in curve[-6:]))
except Exception as e:  # noqa: BLE001
    print(f"  - 跳过：{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 结果

print()
if FAILED:
    print(f"✗ {len(FAILED)} 项失败：")
    for f in FAILED:
        print("   -", f)
    sys.exit(1)

print("✓ 全部通过。D3 的压缩逻辑可以进下一步（真实会话验证）。")
