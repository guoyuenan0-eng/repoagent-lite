# src/repoagent/paths.py
from __future__ import annotations

import os
from pathlib import Path

# 唯一的真相源。所有工具、编辑、检索都从这里取
REPO_ROOT = Path(os.getenv("REPOAGENT_ROOT", ".")).resolve()


class PathError(ValueError):
    """路径不合法（越界 / 空 / 类型错）。"""


def resolve_in_repo(path: str, *, must_exist: bool = False,
                    must_be_dir: bool = False) -> Path:
    """把模型给的路径归一化成仓库内的绝对路径。

    三个职责，且只在这里做一次：
      1. 清洗：去反引号/引号/空格/dot 前缀
      2. 越界检查：最终路径必须在 REPO_ROOT 之内
      3. 存在性检查（可选）
    """
    if not isinstance(path, str):
        raise PathError(f"path 必须是字符串，收到 {type(path).__name__}")

    clean = path.strip().strip("`'\"").strip().lstrip("./") or "."
    p = (REPO_ROOT / clean).resolve()

    # is_relative_to 比 `REPO_ROOT in p.parents` 更准：它把 REPO_ROOT 自身也算作合法
    if p != REPO_ROOT and not p.is_relative_to(REPO_ROOT):
        raise PathError(f"路径越界: {path!r} → {p}（只能访问仓库内文件）")

    if must_exist and not p.exists():
        parent = p.parent if p.parent.exists() else REPO_ROOT
        raise FileNotFoundError(
            f"文件不存在: {relative(p)}。该目录下有: {_preview(parent)}"
        )
    if must_be_dir and not p.is_dir():
        raise NotADirectoryError(f"{relative(p)} 不是目录")
    return p


def relative(p: Path) -> str:
    """统一用仓库相对路径对外展示，避免把绝对路径喂给模型。"""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _preview(d: Path, limit: int = 30) -> str:
    try:
        return str(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())[:limit])
    except OSError:
        return "（无法读取）"