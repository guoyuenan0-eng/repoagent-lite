"""D2 五阶段调度。"""
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from repoagent.llm import LLM, TruncatedToolCallError
from repoagent.state import ToolCall, ToolResult, StopReason, SessionState, Message, Status, Step
from repoagent.tool import ToolRegistry, Tool, read_file
import logging

log = logging.getLogger(__name__)
REPO_ROOT = Path(os.getenv("REPOAGENT_ROOT", ".")).resolve()
registry = ToolRegistry()
registry.register(Tool(
    name="read_file",
    # 描述是 prompt 的一部分，不是文档。要写清：干什么、什么时候不该用、路径基准
    description=(
        "读取仓库中某个文件的文本内容。"
        "只用于读取；修改文件请用 edit_file。"
        "path 必须相对于仓库根目录，例如 src/main.py。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对于仓库根目录的文件路径",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    func=read_file,
))


def build_context(messages):
    return messages

def execute_tools(calls: list[ToolCall], registry: ToolRegistry):
    result = []
    for tool_call in calls:
        name = tool_call.name
        args = tool_call.args
        try:
            tool = registry.get(name)
            if tool is None:
                raise Exception(f"Tool {name} not found")

            out = tool.func(**args)
            result.append(ToolResult(
                tool_call_id=tool_call.id,
                ok=True,
                content=str(out)
            ))
        except Exception as e:
            result.append(ToolResult(
                tool_call_id=tool_call.id,
                ok=False,
                error=f"{type(e).__name__}: {e}"
            ))
    return result

def check_stop(state):
    b = state.budget

    # 1. 步数
    if b.steps_used >= b.max_steps:
        return StopReason.MAX_STEPS

    # 2. 预算
    if b.tokens_used >= b.max_tokens or b.usd_used >= b.max_usd:
        return StopReason.BUDGET_EXCEEDED

    # 3. 产物有效
    for meta in state.artifacts.values():
        if meta.validated and meta.sha256:
            return StopReason.ARTIFACT_VALID

    # 4. 无进展：最近 3 步都没有任何工具调用
    recent = state.steps[-3:]
    if len(recent) == 3 and all(not s.tool_calls for s in recent):
        return StopReason.NO_PROGRESS
    return StopReason.NONE

def run(state: SessionState, task: str, llm: LLM,
        trace_path: str = "traces/run.jsonl") -> StopReason:
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    trace = open(trace_path, "a", encoding="utf-8")
    state.messages.append(Message(role="user", content=task))
    try:
        while True:
            reason = check_stop(state)
            if reason is not StopReason.NONE:
                state.status = Status.DONE if reason is StopReason.ARTIFACT_VALID else Status.FAILED
                return reason
            ctx = build_context(state.messages)
            t0 = time.monotonic()
            try:
                msg, usage = llm.call_model(ctx, registry.schemas())
            except TruncatedToolCallError as e:
                state.status = Status.FAILED
                return StopReason.ERROR
            except Exception as e:
                # 网络/限流/400：本轮失败，上层决定是否重试
                state.status = Status.FAILED
                return StopReason.ERROR
            text, calls = llm.parse_output(msg)
            latency = int((time.monotonic() - t0) * 1000)
            tokens_in = getattr(usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(usage, "completion_tokens", 0) or 0
            state.budget.spend(tokens_in + tokens_out, 0.0)
            state.messages.append(Message(role="assistant", content=text, tool_calls=calls))
            step = Step(
                index=len(state.steps),
                assistant_text=text[:500],
                tool_calls=calls,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency,
            )
            # 模型不再要工具 => 自然结束
            if not calls:
                state.steps.append(step)
                trace.write(json.dumps(step.model_dump(), ensure_ascii=False) + "\n")
                trace.flush()
                state.status = Status.DONE
                return StopReason.NONE
            results = execute_tools(calls, registry)
            for r in results:
                state.messages.append(Message(
                    role="tool",
                    content=r.content if r.ok else (r.error or ""),
                    tool_call_id=r.tool_call_id,
                ))
            step.tool_results = results
            state.steps.append(step)
            trace.write(json.dumps(step.model_dump(), ensure_ascii=False) + "\n")
            trace.flush()
    finally:
        trace.close()

if __name__ == "__main__":
    load_dotenv()
    llm = LLM()
    state = SessionState(session_id=str(uuid.uuid4())[:8])
    state.budget.max_steps = 5
    reason = run(state, "读一下 README.md，然后用一句话告诉我这个项目是干什么的。", llm)
    print(f"\nstop_reason = {reason}, status = {state.status}")
    print(f"steps = {len(state.steps)}, tokens = {state.budget.tokens_used}")
    for s in state.steps:
        names = [c.name for c in s.tool_calls]
        print(f"  step {s.index}: tools={names or '-'} "
              f"in={s.tokens_in} out={s.tokens_out} {s.latency_ms}ms")