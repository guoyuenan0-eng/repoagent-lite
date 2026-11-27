"""D5 · 编辑工具测试（先写，全部应为红）
契约（实现必须满足）：
    · 编辑函数**不抛异常**，失败一律返回 EditResult(ok=False, error=...)
    · 三个函数都走 resolve_in_repo()，因此 monkeypatch paths.REPO_ROOT 就能隔离到临时目录
    · 任何一个失败都必须让文件保持原样（回滚 / 原子写）
跑法：
    cp .../tests_test_editing.py tests/test_editing.py
    pytest tests/test_editing.py -v
"""
from __future__ import annotations
import pytest
import repoagent.paths as paths
from repoagent.editing import apply_patch, replace_in_file, write_file
@pytest.fixture
def repo(tmp_path, monkeypatch):
    """把仓库根目录临时指向 tmp_path，测试之间完全隔离。"""
    root = tmp_path.resolve()
    monkeypatch.setattr(paths, "REPO_ROOT", root)
    return root
def make(repo, name: str, text: str):
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p
# ================================================================
# replace_in_file
# ================================================================
def test_replace_happy_path(repo):
    f = make(repo, "m.py", "def add(a, b):\n    return a - b\n")
    r = replace_in_file("m.py", "return a - b", "return a + b")
    assert r.ok, r.error
    assert "a + b" in f.read_text(encoding="utf-8")
    assert r.lines_changed >= 1
def test_replace_rejects_ambiguous_match(repo):
    """出现多次必须拒绝 —— 这是「改坏代码」的头号原因。"""
    f = make(repo, "m.py", "x = 1\nx = 1\nx = 1\n")
    r = replace_in_file("m.py", "x = 1", "x = 2")
    assert not r.ok
    assert "3" in r.error, f"错误信息应说明匹配到几处，实际: {r.error}"
    assert f.read_text(encoding="utf-8") == "x = 1\nx = 1\nx = 1\n"
def test_replace_rejects_missing_match(repo):
    f = make(repo, "m.py", "x = 1\n")
    r = replace_in_file("m.py", "x = 999", "x = 2")
    assert not r.ok
    assert f.read_text(encoding="utf-8") == "x = 1\n"
def test_replace_rejects_empty_old_string(repo):
    """空 old_string 会匹配到任何位置，必须显式拒绝。"""
    make(repo, "m.py", "x = 1\n")
    r = replace_in_file("m.py", "", "x = 2")
    assert not r.ok
def test_replace_with_empty_new_deletes(repo):
    f = make(repo, "m.py", "a = 1\nb = 2\n")
    r = replace_in_file("m.py", "b = 2\n", "")
    assert r.ok, r.error
    assert f.read_text(encoding="utf-8") == "a = 1\n"
def test_replace_rolls_back_on_broken_syntax(repo):
    """改完语法错误 → 拒绝写入，原文件一个字节都不能变。"""
    f = make(repo, "m.py", "x = 1\n")
    r = replace_in_file("m.py", "x = 1", "x = (")
    assert not r.ok
    assert "语法" in r.error or "SyntaxError" in r.error
    assert f.read_text(encoding="utf-8") == "x = 1\n"
def test_replace_skips_syntax_check_for_non_python(repo):
    """非 .py 文件不做语法校验（markdown / json 会被误判）。"""
    f = make(repo, "README.md", "# x = 1\n")
    r = replace_in_file("README.md", "x = 1", "x = (")
    assert r.ok, r.error
def test_replace_leaves_no_temp_file(repo):
    make(repo, "m.py", "x = 1\n")
    replace_in_file("m.py", "x = 1", "x = 2")
    leftovers = [p.name for p in repo.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == [], f"残留临时文件: {leftovers}"
def test_replace_missing_file(repo):
    r = replace_in_file("nope.py", "a", "b")
    assert not r.ok
# ================================================================
# apply_patch
# ================================================================
def test_apply_patch_single_hunk(repo):
    f = make(repo, "m.py", "a = 1\nb = 2\nc = 3\n")
    patch = (
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 20\n"
        " c = 3\n"
    )
    r = apply_patch("m.py", patch)
    assert r.ok, r.error
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 20\nc = 3\n"
def test_apply_patch_multiple_hunks(repo):
    f = make(repo, "m.py", "a = 1\nb = 2\nc = 3\nd = 4\n")
    patch = (
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-a = 1\n"
        "+a = 10\n"
        " b = 2\n"
        "@@ -3,2 +3,2 @@\n"
        " c = 3\n"
        "-d = 4\n"
        "+d = 40\n"
    )
    r = apply_patch("m.py", patch)
    assert r.ok, r.error
    assert f.read_text(encoding="utf-8") == "a = 10\nb = 2\nc = 3\nd = 40\n"
def test_apply_patch_tolerates_wrong_line_numbers(repo):
    """行号写错但上下文唯一 → 仍应成功（靠上下文定位，不靠行号）。"""
    f = make(repo, "m.py", "a = 1\nb = 2\nc = 3\n")
    patch = (
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -100,3 +100,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 20\n"
        " c = 3\n"
    )
    r = apply_patch("m.py", patch)
    assert r.ok, f"行号不该影响定位: {r.error}"
    assert "b = 20" in f.read_text(encoding="utf-8")
def test_apply_patch_rejects_context_mismatch(repo):
    f = make(repo, "m.py", "a = 1\nb = 2\n")
    patch = (
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -1,2 +1,2 @@\n"
        " a = 1\n"
        "-zzz = 999\n"
        "+zzz = 0\n"
    )
    r = apply_patch("m.py", patch)
    assert not r.ok
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 2\n"
def test_apply_patch_rejects_garbage(repo):
    make(repo, "m.py", "a = 1\n")
    r = apply_patch("m.py", "这不是一个 patch")
    assert not r.ok
def test_apply_patch_rolls_back_on_broken_syntax(repo):
    f = make(repo, "m.py", "a = 1\nb = 2\n")
    patch = (
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-a = 1\n"
        "+a = (\n"
        " b = 2\n"
    )
    r = apply_patch("m.py", patch)
    assert not r.ok
    assert f.read_text(encoding="utf-8") == "a = 1\nb = 2\n"
# ================================================================
# write_file（只用于新建）
# ================================================================
def test_write_creates_new_file(repo):
    r = write_file("new.py", "x = 1\n")
    assert r.ok, r.error
    assert (repo / "new.py").read_text(encoding="utf-8") == "x = 1\n"
def test_write_creates_parent_dirs(repo):
    r = write_file("a/b/c/new.py", "x = 1\n")
    assert r.ok, r.error
    assert (repo / "a/b/c/new.py").exists()
def test_write_refuses_to_overwrite(repo):
    """默认不允许覆盖 —— 整文件重写会丢掉未提及的代码。"""
    f = make(repo, "m.py", "x = 1\ny = 2\n")
    r = write_file("m.py", "x = 999\n")
    assert not r.ok
    assert f.read_text(encoding="utf-8") == "x = 1\ny = 2\n"
def test_write_rejects_broken_syntax(repo):
    r = write_file("bad.py", "def f(:\n")
    assert not r.ok
    assert not (repo / "bad.py").exists()
# ================================================================
# 路径安全（三个函数共用 resolve_in_repo，验证一次即可）
# ================================================================
@pytest.mark.parametrize("bad", ["../escape.py", "/etc/passwd", "a/../../x.py"])
def test_path_escape_rejected(repo, bad):
    r = replace_in_file(bad, "a", "b")
    assert not r.ok
    assert "越界" in (r.error or "") or "PathError" in (r.error or "") or r.error
def test_backticks_are_stripped(repo, ):
    f = make(repo, "m.py", "x = 1\n")
    r = replace_in_file("`m.py`", "x = 1", "x = 2")
    assert r.ok, f"模型常把路径包在反引号里: {r.error}"
    assert "x = 2" in f.read_text(encoding="utf-8")