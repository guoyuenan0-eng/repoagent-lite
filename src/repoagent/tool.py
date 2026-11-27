from pathlib import Path
from typing import Callable
from dataclasses import dataclass

from repoagent.paths import resolve_in_repo, relative
from repoagent import paths

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
    p = resolve_in_repo(path, must_be_dir=True)
    return _preview_dir(p)


def read_file(path: str) -> str:
    p = resolve_in_repo(path)
    if p.is_dir():
        raise IsADirectoryError(f"{relative(p)} 是目录，用 list_dir")
    text = p.read_text(encoding="utf-8", errors="replace")
    return text