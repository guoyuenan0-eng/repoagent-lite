import ast
import re
from dataclasses import dataclass

from repoagent.paths import resolve_in_repo, PathError


@dataclass
class EditResult:
    ok: bool
    strategy: str          # replace / patch / write
    error: str | None = None
    lines_changed: int = 0

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

def _parse_hunks(patch: str):
    """把 patch 文本解析成 hunks 列表。
    每个 hunk: {'old_start': int, 'lines': [(op, text), ...]}
    op 取值: ' ' 上下文, '-' 删除, '+' 新增
    """
    hunks = []
    cur = None
    for raw in patch.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            cur = {"old_start": int(m.group(1)), "lines": []}
            hunks.append(cur)
            continue
        if cur is None:
            continue  # 文件头 ---/+++ 等忽略
        if not raw:
            # 空行在 diff 里通常是空上下文行，按 ' ' 处理
            cur["lines"].append((" ", ""))
            continue
        op, text = raw[0], raw[1:]
        if op in (" ", "-", "+"):
            cur["lines"].append((op, text))
        # 其他（如 '\ No newline'）忽略
    return hunks

def _apply_hunk(lines: list[str], hunk: dict) -> list[str]:
    """把一个 hunk 应用到 lines，返回新 lines。"""
    old_block = [t for op, t in hunk["lines"] if op in (" ", "-")]
    new_block = [t for op, t in hunk["lines"] if op in (" ", "+")]

    if not old_block:
        # 纯新增 hunk，按 old_start 附近插入
        start = min(max(hunk["old_start"] - 1, 0), len(lines))
        return lines[:start] + new_block + lines[start:]

    matches = [
        i for i in range(len(lines) - len(old_block) + 1)
        if lines[i:i + len(old_block)] == old_block
    ]
    if not matches:
        raise ValueError("patch 上下文在文件中找不到匹配")
    if len(matches) > 1:
        # 多处匹配时，选离 old_start 最近的
        start = min(matches, key=lambda i: abs(i - (hunk["old_start"] - 1)))
    else:
        start = matches[0]
    return lines[:start] + new_block + lines[start + len(old_block):]

def replace_in_file(path: str, old: str, new: str) -> EditResult:
    try:
        p = resolve_in_repo(path, must_exist=True)  # 编辑必须目标存在
    except Exception as e:
        return EditResult(False, "replace", str(e))

    original = p.read_text(encoding="utf-8")
    # 唯一匹配
    count = original.count(old)
    if count == 0:
        return EditResult(False, "replace", "old_string 未找到")
    if count > 1:
        return EditResult(False, "replace", f"old_string 出现 {count} 次，请提供更多上下文")

    updated = original.replace(old, new)

    # 语法校验（Python 用 ast，不需要 tree-sitter）
    if path.endswith(".py"):
        try:
            ast.parse(updated)
        except SyntaxError as e:
            return EditResult(False, "replace", f"改后语法错误: {e}")

    # 原子写：先写临时文件再 rename，避免半截文件
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(updated, encoding="utf-8")
    tmp.replace(p)
    return EditResult(True, "replace", None, new.count("\n") + 1)


def apply_patch(path: str, patch: str) -> EditResult:
    try:
        p = resolve_in_repo(path, must_exist=True)  # 编辑必须目标存在
    except Exception as e:
        return EditResult(False, "patch", str(e))

    hunks = _parse_hunks(patch)
    if not hunks:  # ← 新增：解析不到就明确失败
        return EditResult(False, "patch", "没有解析到任何 @@ 改动块，请检查 patch 格式")

    original = p.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in original else "\n"
    lines = original.splitlines()
    for hunk in hunks:
        try:
            lines = _apply_hunk(lines, hunk)
        except ValueError as e:
            return EditResult(False, "patch", "patch 上下文在文件中找不到匹配")

    updated = newline.join(lines)
    if original.endswith(("\n", "\r\n")):
        updated += newline  # 保留原文件末尾换行习惯

    if updated == original:
        return EditResult(False, "patch", "patch 已解析，但应用后文件内容无变化")  # 无变化，不写盘

    if p.suffix == ".py":
        try:
            ast.parse(updated)
        except SyntaxError as e:
            return EditResult(False, "write", f"语法错误: {e}")

    # 原子写：先写临时文件再 rename，避免半截文件
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(updated, encoding="utf-8")
    tmp.replace(p)
    return EditResult(True, "patch", None, 0)

def write_file(path: str, content: str, overwrite: bool = False) -> EditResult:
    p = resolve_in_repo(path)
    if p.exists() and p.is_dir():
        return EditResult(False, "write", "目标是目录")
    if p.exists() and not overwrite:
        return EditResult(False, "write", "文件已存在，请用 edit_file / apply_patch 精确修改")

    if p.suffix == ".py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            return EditResult(False, "write", f"语法错误: {e}")

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")  # 原子写，避免半截文件
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(p)
    return EditResult(True, "write", lines_changed=len(content.splitlines()))