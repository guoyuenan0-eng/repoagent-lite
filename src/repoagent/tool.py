from pathlib import Path
from typing import Callable
from dataclasses import dataclass

REPO_ROOT = Path(".").resolve()
MAX_FILE_BYTES = 200_000
MAX_OUTPUT_CHARS = 20_000

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def to_schema(self) -> dict:
        """转成 provider 要的 tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

class ToolRegistry:
    def __init__(self):
        self._tools : dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name not in self._tools:
            self._tools[tool.name] = tool
        else:
            raise ValueError(f"tool name already registered: {tool.name}")

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)


def _preview_dir(d: Path, limit: int = 30) -> str:
    try:
        names = sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())
    except OSError:
        return "（无法读取）"
    return str(names[:limit])


def list_dir(path: str = ".") -> str:
    """列目录。给 agent 一个「先看有什么」的手段，避免盲猜文件名。"""
    clean = (path or ".").strip().strip("`'\"").strip() or "."
    p = (REPO_ROOT / clean).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"路径越界: {clean}")
    if not p.is_dir():
        raise NotADirectoryError(f"{clean} 不是目录")
    return _preview_dir(p)


def read_file(path: str) -> str:
    clean = (path or "").strip().strip("`'\"").strip()
    p = (REPO_ROOT / clean).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"路径越界: {clean}（只能访问仓库内文件）")
    if p.is_dir():
        raise IsADirectoryError(f"{clean} 是目录不是文件。目录内容: {_preview_dir(p)}")
    if not p.exists():
        # 关键：报错里给出同目录内容，模型才能自我纠正
        # 裸的 FileNotFoundError 等于让模型继续猜
        parent = p.parent if p.parent.exists() else REPO_ROOT
        raise FileNotFoundError(f"文件不存在: {clean}。该目录下有: {_preview_dir(parent)}")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file too large: {p.stat().st_size} bytes")
    text = p.read_text(encoding="utf-8", errors="replace")
    return text