# tests/test_retrieval.py
from repoagent.retrieval import repo_map, grep, search_symbol

def test_repo_map_fits_budget():
    m = repo_map(".")
    assert len(m) < 8000, "map 必须是骨架，不能是全文"
    assert "class Budget" in m
    assert "class SessionState" in m
    assert "def pack_tool_outputs" in m

def test_grep_finds_exact():
    hits = grep("pack_tool_outputs")
    assert "manager.py" in hits

def test_search_symbol_finds_definition():
    r = search_symbol("StopReason")
    assert "state.py" in r
    assert "class StopReason" in r