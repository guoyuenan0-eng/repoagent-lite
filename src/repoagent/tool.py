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


def read_file(path: str) -> str:
    p = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError(f"path escapes repo root: {path}")
    if not p.exists():
        raise FileNotFoundError(path)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"file too large: {p.stat().st_size} bytes")
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:MAX_OUTPUT_CHARS]