"""D2 五阶段调度。"""
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from repoagent.editing import replace_in_file, apply_patch, write_file
from repoagent.llm import LLM, TruncatedToolCallError
from repoagent.retrieval import repo_map, grep, search_symbol
from repoagent.state import ToolCall, ToolResult, StopReason, SessionState, Message, Status, Step
from repoagent.tool import ToolRegistry, Tool, read_file, list_dir
from repoagent.context.manager import *
from repoagent.context.manager import build_context as _build_ctx
from repoagent import paths
import logging

log = logging.getLogger(__name__)
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
registry.register(Tool(
    name="list_dir",
    description=(
        "列出某个目录下的文件和子目录。"
        "不知道文件放在哪、或 read_file 报「文件不存在」时用它确认路径。"
        "path 相对于仓库根目录，根目录用 \".\"。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对仓库根目录的目录路径，默认 ."},
        },
        "required": [],
        "additionalProperties": False,
    },
    func=lambda path=".": list_dir(path),
))

registry.register(Tool(
    name="repo_map",
    description=(
        "返回整个仓库的代码骨架：每个文件下的类、函数签名和行号，不含函数体。"
        "第一次接触仓库、需要了解它有哪几个模块时先调它。"
        "它只有签名没有实现，不要用它代替 read_file 去查具体逻辑。"
    ),
    parameters={"type": "object", "properties": {}, "required": []},
    func=lambda: repo_map(str(paths.REPO_ROOT)),
))

registry.register(Tool(
    name="search_symbol",
    description=(
        "按符号名查找类 / 函数的**定义位置**，返回 路径:行号: class 或 def 名字。"
        "只知道符号名、不知道在哪个文件时用它。"
        "它只找定义；找谁调用过某个符号，请用 grep。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "符号名或名字的一部分，例如 StopReason、build_context",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    func=lambda name: search_symbol(name, root=str(paths.REPO_ROOT)),
))

registry.register(Tool(
    name="grep",
    description=(
        "在整个仓库里按正则表达式搜索文本，返回 路径:行号: 内容。"
        "用于查找已知字符串：函数名、常量、报错信息、配置项。"
        "代码里的名字是精确的——想找某个名字就先用它，不要凭记忆猜文件。"
        "pattern 是 Python 正则；只想找定义请用 search_symbol。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Python 正则表达式，例如 build_context、def run\\(、TODO",
            }
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    func=lambda pattern: grep(pattern, root=str(paths.REPO_ROOT)),
))


registry.register(Tool(
    name="edit_file",
    description=(
        "把文件里的一段文本精确替换成另一段。这是**修改已有文件的首选方式**。\n"
        "old_string 必须在文件中**唯一匹配**——如果这段内容出现多次，"
        "请多带几行上下文（例如把函数签名和它上面一行一起带上），"
        "否则工具会拒绝执行并告诉你匹配到了几处。\n"
        "old_string 要原样复制（含缩进），不要省略号、不要改写空白。\n"
        "需要改很多处分散位置时用 apply_patch；新建文件用 create_file。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对仓库根目录的文件路径，例如 src/repoagent/state.py",
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的原文本，必须与文件内容逐字符一致（含缩进）",
            },
            "new_string": {
                "type": "string",
                "description": "替换后的文本；留空字符串表示删除这段",
            },
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    },
    func=lambda path, old_string, new_string: str(
        replace_in_file(path, old_string, new_string)
    ),
))

registry.register(Tool(
    name="apply_patch",
    description=(
        "把一个 unified diff（和 git diff 同格式）应用到某个文件。\n"
        "**当同一文件有多处分散的小改动时用它**，比 edit_file 省大量 token。\n"
        "格式要求：以 --- / +++ 开头，每个改动块以 @@ 开头；"
        "每个块的上下文行前面有一个空格，删除行前面是 -，新增行前面是 +。\n"
        "每处改动建议带 3 行上下文——行号写不精确没关系（工具靠上下文定位），"
        "但上下文必须能唯一确定位置。\n"
        "只改一处、或改动很小时用 edit_file 更稳。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对仓库根目录的文件路径",
            },
            "patch": {
                "type": "string",
                "description": "标准 unified diff 文本，可包含多个 @@ 块",
            },
        },
        "required": ["path", "patch"],
        "additionalProperties": False,
    },
    func=lambda path, patch: str(apply_patch(path, patch)),
))

registry.register(Tool(
    name="create_file",
    description=(
        "新建一个文件并写入内容。**只能用于目标文件尚不存在的情况。**\n"
        "如果文件已存在，工具会拒绝——修改已有文件请用 edit_file 或 apply_patch，"
        "因为整文件重写会丢掉你没提到的注释和其他代码。\n"
        "父目录不存在时会自动创建。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "相对仓库根目录的文件路径，例如 src/repoagent/editing/patch.py",
            },
            "content": {
                "type": "string",
                "description": "文件的完整内容",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    func=lambda path, content: str(write_file(path, content)),
))



def build_context(messages):
    return _build_ctx(messages)

def execute_tools(calls: list[ToolCall], registry: ToolRegistry):
    result = []
    for tool_call in calls:
        name = tool_call.name
        args = tool_call.args
        try:
            tool = registry.get(name)
            if tool is None:
                raise ValueError(f"unknown tool: {tool_call.name}; available={registry.names()}")

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
    print(registry.names())
    load_dotenv()
    llm = LLM()
    state = SessionState(session_id=str(uuid.uuid4())[:8])
    state.budget.max_steps = 5
    reason = run(state, "StopReason 定义在哪？哪些文件用了它？", llm)
    print(f"\nstop_reason = {reason}, status = {state.status}")
    print(f"steps = {len(state.steps)}, tokens = {state.budget.tokens_used}")
    for s in state.steps:
        names = [c.name for c in s.tool_calls]
        print(f"  step {s.index}: tools={names or '-'} "
              f"in={s.tokens_in} out={s.tokens_out} {s.latency_ms}ms")