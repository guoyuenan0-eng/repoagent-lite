import ast
import re
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "traces", ".idea", ".mypy_cache", ".pytest_cache", "build", "dist",
    ".eggs",
}
SKIP_SUFFIXES = {".egg-info"}

@dataclass
class Symbol:
    kind: str      # class / def / method
    name: str
    args: str
    path: str
    lineno: int

def _is_skipped(p: Path) -> bool:
    if any(part in SKIP_DIRS for part in p.parts):
        return True
    return any(part.endswith(".egg-info") for part in p.parts)

def iter_python_files(root: str | Path = ".") -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*.py") if not _is_skipped(p))

def grep(pattern: str, root: str | Path = ".", max_hits: int = 40) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"正则非法: {e}"
    hits: list[str] = []
    for p in iter_python_files(root):
        try:
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    rel = str(p.relative_to(Path(root)))
                    hits.append(f"{rel}:{i}: {line.strip()[:120]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits) + f"\n...（已截断，超过 {max_hits} 条）"
        except OSError:
            continue
    return "\n".join(hits) if hits else f"无匹配: {pattern}"

def _signature(fn: ast.FunctionDef) -> str:
    """把参数拼成 `a, b, c=...` 形式，只保留名字，忽略类型注解细节。"""
    a = fn.args
    names = [x.arg for x in a.args]
    ndef = len(a.defaults)
    for i in range(len(names) - ndef, len(names)):
        names[i] = f"{names[i]}=..."
    if a.vararg:
        names.append(f"*{a.vararg.arg}")
    if a.kwarg:
        names.append(f"**{a.kwarg.arg}")
    return ", ".join(names)

def extract_symbols(path: Path, root: Path) -> list[Symbol]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, UnicodeDecodeError):
        return []            # 坏文件不该让整个索引挂掉
    rel = str(path.relative_to(root))
    out: list[Symbol] = []
    for node in tree.body:                       # 只看顶层
        if isinstance(node, ast.ClassDef):
            out.append(Symbol("class", node.name, "", rel, node.lineno))
            for sub in node.body:                # 类内方法
                if isinstance(sub, ast.FunctionDef):
                    out.append(Symbol("method", sub.name, _signature(sub), rel, sub.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Symbol("def", node.name, _signature(node), rel, node.lineno))
    return out

def build_index(root: str | Path = ".") -> list[Symbol]:
    root = Path(root)
    syms: list[Symbol] = []
    for p in iter_python_files(root):
        syms.extend(extract_symbols(p, root))
    return syms

def search_symbol(name: str, root: str | Path = ".") -> str:
    """按符号名找定义行（精确名前缀匹配，不区分大小写）。"""
    hits = [s for s in build_index(root) if name.lower() in s.name.lower()]
    if not hits:
        return f"未找到符号: {name}"
    lines = []
    for s in hits[:20]:
        kw = "class" if s.kind == "class" else "def"
        lines.append(f"{s.path}:{s.lineno}  {kw} {s.name}({s.args})")
    return "\n".join(lines)

def repo_map(root: str | Path = ".", max_chars: int = 8000) -> str:
    """仓库骨架：文件 → 顶层类/函数签名。硬截断到 max_chars。"""
    root = Path(root)
    blocks: list[str] = []
    used = 0
    # 按文件大小排序：小文件（通常是核心模块）优先放进预算
    files = iter_python_files(root)
    for p in files:
        syms = extract_symbols(p, root)
        if not syms:
            continue
        rel = str(p.relative_to(root))
        lines = [rel]
        for s in syms:
            if s.kind == "class":
                lines.append(f"  class {s.name}  (L{s.lineno})")
            elif s.kind == "method":
                lines.append(f"    def {s.name}({s.args})")
            else:
                lines.append(f"  def {s.name}({s.args})  (L{s.lineno})")
        block = "\n".join(lines)
        if used + len(block) > max_chars:
            blocks.append(f"...（已截断，共 {len(files)} 个文件）")
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print("=" * 60)
    print("REPO MAP")
    print("=" * 60)
    m = repo_map(root)
    print(m)
    print(f"\n[{len(m)} 字符]")
    print("\n" + "=" * 60)
    print("search_symbol('StopReason')")
    print("=" * 60)
    print(search_symbol("StopReason", root))
    print("\n" + "=" * 60)
    print("grep('pack_tool_outputs')")
    print("=" * 60)
    print(grep("pack_tool_outputs", root))