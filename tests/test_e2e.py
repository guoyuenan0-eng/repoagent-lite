"""D3 端到端验收：真实会话里压缩会不会生效，以及压完还能不能答对

跑法（在 ~/code/repoagent-lite 下）：
    source .venv/bin/activate
    python .../notes/03-e2e.py --fake     # 不花钱，验证接线
    python .../notes/03-e2e.py --live     # 真实模型，验证压缩后仍能答对

它做的事：
    1. 造 20 个各 4000 字符的文件（每个带唯一标记）
    2. 让 agent 依次读完，最后回答第 1 个文件的标记
    3. 统计：压缩触发次数、步数、最终回答是否正确

为什么任务要问"第 1 个文件的标记"：
    压缩会把最早的内容折叠掉。如果模型还能答对，说明要么没被压到，
    要么它自己重新读了 —— 两种都是可接受的行为；答错则说明压缩策略
    丢了任务必需的信息（那就需要"产物全文保留/引用式回读"）。
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS

from dotenv import load_dotenv

import repoagent.tool as T
from repoagent import loop
from repoagent.context import manager as M
from repoagent.state import SessionState

N_FILES = 20
FILE_CHARS = 4000


def build_fake_repo(root: Path) -> None:
    (root / "data").mkdir(parents=True, exist_ok=True)
    for i in range(N_FILES):
        body = f"MARKER_f{i:02d}\n" + ("填充内容。" * (FILE_CHARS // 5))
        (root / "data" / f"f{i:02d}.txt").write_text(body, encoding="utf-8")


class ScriptedLLM(loop.LLM):
    """不用 API 的脚本化模型：依次读文件，最后给答案。

    继承 LLM 以复用 parse_output / _loads_lenient，但**不调用 super().__init__**，
    避免真的去建 OpenAI client。
    """

    def __init__(self, n_files: int):
        self.n_files = n_files
        self.i = 0
        self.last_payload = None

    def call_model(self, messages, tools=None):
        if self.i < self.n_files:
            k = self.i
            self.i += 1
            msg = NS(content="", tool_calls=[NS(
                id=f"c{k}",
                function=NS(name="read_file", arguments=f'{{"path": "data/f{k:02d}.txt"}}'),
            )])
        else:
            msg = NS(content="MARKER_f00", tool_calls=None)
        return msg, NS(prompt_tokens=500, completion_tokens=20)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="用真实模型（花钱）")
    ap.add_argument("--fake", action="store_true", help="用脚本化模型（默认，不花钱）")
    ap.add_argument("--keep", action="store_true", help="保留临时仓库")
    args = ap.parse_args()

    import shutil
    import tempfile
    import time as _time

    tmp = Path(tempfile.mkdtemp(prefix="rl_e2e_"))
    build_fake_repo(tmp)

    # trace 显式指定，写在仓库的 traces/ 下（已 gitignore），不要藏在临时目录里
    trace_path = Path("traces") / f"e2e_{_time.strftime('%m%d_%H%M%S')}.jsonl"
    print(f"[trace] {trace_path.resolve()}")

    # 让 read_file 指向假仓库（REPO_ROOT 是模块级变量，直接改）
    T.REPO_ROOT = tmp.resolve()   # macOS 的 /var -> /private/var 软链，必须 resolve

    # 统计压缩次数
    stats = {"compressed": 0, "calls": 0}
    real_compress = M.compress

    def counting_compress(messages, keep_recent=10):
        out = real_compress(messages, keep_recent=keep_recent)
        stats["compressed"] += 1
        return out

    M.compress = counting_compress
    loop._build_ctx.__globals__["compress"] = counting_compress  # type: ignore[attr-defined]

    if args.live:
        llm = loop.LLM()
    else:
        llm = ScriptedLLM(N_FILES)

    state = SessionState(session_id="e2e")
    state.budget.max_steps = 40

    # task = (
    #     f"依次读取 data/ 目录下 f00.txt 到 f{N_FILES - 1:02d}.txt 的全部文件，"
    #     "读完后只回答 f00.txt 的第一个标记（形如 MARKER_xxx），不要解释。"
    # )

    task = "StopReason 定义在哪？哪些文件用了它？"

    buf = io.StringIO()
    with redirect_stdout(buf):
        reason = loop.run(state, task, llm, trace_path=str(trace_path))
    log = buf.getvalue()

    # ---------------------------------------------------------------- 报告
    answer = ""
    for s in state.steps:
        if not s.tool_calls:
            answer = s.assistant_text or answer

    print("=" * 56)
    print(f"停止原因   : {reason}")
    print(f"步数       : {len(state.steps)}")
    print(f"压缩次数   : {stats['compressed']}")
    n_calls = sum(len(s.tool_calls) for s in state.steps)
    n_ok = sum(1 for s in state.steps for r in s.tool_results if r.ok)
    print(f"工具调用   : {n_calls} 次，成功 {n_ok} 次")
    print(f"最终回答   : {answer[:80]!r}")
    n_lines = sum(1 for _ in trace_path.open(encoding="utf-8")) if trace_path.exists() else 0
    print(f"trace 行数 : {n_lines}  ({trace_path})")
    print(f"答对 f00   : {'是' if 'MARKER_f00' in answer else '否'}")
    print("=" * 56)
    if "[压缩]" in log:
        for line in log.splitlines():
            if "[压缩]" in line:
                print(line.strip())

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"临时仓库保留在: {tmp}")

    # ---------------------------------------------------------------- 判定
    ok = True
    if stats["compressed"] == 0:
        print("✗ 压缩从未触发：检查预算/阈值，或文件太小")
        ok = False
    if args.live and "MARKER_f00" not in answer:
        print("⚠ 模型没答对第一个文件的标记 —— 压缩可能把必需信息丢了")
        print("  修法：折叠时保留文件路径 + 首个非空行，别只记「调用过」")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
