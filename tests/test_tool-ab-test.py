"""D4 · A/B 验收：有检索 vs 没检索，定位成本差多少

跑法（在 ~/code/repoagent-lite 下）：
    source .venv/bin/activate
    python .../notes/04-ab-test.py --live

它做两次同样的任务：
    A 组：只有 read_file和list_dir（现状基线）
    B 组：read_file + repo_map + search_symbol + grep

量化的四件事：工具调用数、步数、token、耗时。
"""
from __future__ import annotations

import sys
import time

from dotenv import load_dotenv

import repoagent.loop as L
from repoagent.state import SessionState
from repoagent.tool import ToolRegistry

TASK = (
    "回答两个问题，只给结论不要解释：\n"
    "1) StopReason 这个类定义在哪个文件的第几行？\n"
    "2) 仓库里哪些文件使用了它（列出文件路径）？"
)

RETRIEVAL_TOOLS = {"repo_map", "search_symbol", "grep"}


def run_once(tools_keep: set[str] | None, label: str):
    """tools_keep=None 表示保留全部工具。"""
    reg = ToolRegistry()
    for name in L.registry.names():
        if tools_keep is None or name in tools_keep:
            reg.register(L.registry.get(name))

    saved = L.registry
    L.registry = reg
    try:
        st = SessionState(session_id=f"ab_{label}")
        st.budget.max_steps = 15
        t0 = time.monotonic()
        reason = L.run(st, TASK, L.LLM(), trace_path=f"traces/ab_{label}_{int(time.time())}.jsonl")
        elapsed = time.monotonic() - t0
    finally:
        L.registry = saved

    calls = sum(len(s.tool_calls) for s in st.steps)
    names = [c.name for s in st.steps for c in s.tool_calls]
    answer = next((s.assistant_text for s in reversed(st.steps) if not s.tool_calls), "")
    return {
        "label": label,
        "tools": L.registry.names() if tools_keep is None else sorted(tools_keep),
        "reason": reason,
        "steps": len(st.steps),
        "calls": calls,
        "tool_names": names,
        "tokens": st.budget.tokens_used,
        "seconds": round(elapsed, 1),
        "answer": answer,
        "hit": "state.py" in answer and "35" in answer,
    }


def main() -> int:
    load_dotenv()
    if "--live" not in sys.argv:
        print("加 --live 才会真的调模型（会花钱，约 10 次调用）")
        return 0

    print("A 组：只有 read_file和list_dir")
    a = run_once({"read_file", "list_dir"}, "baseline")
    print(f"  steps={a['steps']} calls={a['calls']} tokens={a['tokens']} {a['seconds']}s")
    print(f"  tools used: {a['tool_names']}")
    print(f"  answer: {a['answer'][:100]!r}")

    print("\nB 组：加上检索")
    b = run_once(None, "with_retrieval")
    print(f"  steps={b['steps']} calls={b['calls']} tokens={b['tokens']} {b['seconds']}s")
    print(f"  tools used: {b['tool_names']}")
    print(f"  answer: {b['answer'][:100]!r}")

    print("\n" + "=" * 58)
    print(f"{'指标':<12}{'A 无检索':>12}{'B 有检索':>12}{'变化':>12}")
    print("-" * 58)
    for k, name in (("calls", "工具调用"), ("steps", "步数"), ("tokens", "token"), ("seconds", "耗时")):
        va, vb = a[k], b[k]
        pct = f"{(vb - va) / va:+.0%}" if va else "n/a"
        print(f"{name:<12}{va:>12}{vb:>12}{pct:>12}")
    print("=" * 58)
    print(f"答对率: A={'✓' if a['hit'] else '✗'}  B={'✓' if b['hit'] else '✗'}")
    print(f"trace: traces/ab_baseline.jsonl, traces/ab_with_retrieval.jsonl")

    # 判定：检索要么更快，要么更准，否则说明工具描述没写对
    faster = b["calls"] < a["calls"]
    better = b["hit"] and not a["hit"]
    if not (faster or better):
        print("\n⚠ 检索没带来明显收益。优先检查：")
        print("   · repo_map 的描述有没有说清「第一次接触仓库先调它」")
        print("   · grep / search_symbol 的边界是否重叠，导致模型选错")
    return 0

if __name__ == "__main__":
    sys.exit(main())
