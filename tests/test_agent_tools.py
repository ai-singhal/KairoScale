"""Tests for KairoScale.agent.tools."""

import json
from pathlib import Path

from KairoScale.agent.tools import execute_tool, get_agent_tools
from KairoScale.types import ProfileResult


def _makeContext(tmp_path: Path) -> dict:
    profile = ProfileResult(gpu_utilization=75.0, peak_memory_mb=2048.0)
    return {"profile": profile, "repo_path": str(tmp_path)}


def test_getAgentToolsReturnsAllTools():
    profile = ProfileResult()
    tools = get_agent_tools(profile, Path("/tmp"))
    names = [t["name"] for t in tools]
    assert "read_profile" in names
    assert "read_file" in names
    assert "list_files" in names
    assert "search_code" in names
    assert "propose_config" in names


def test_readProfile(tmp_path):
    ctx = _makeContext(tmp_path)
    result = execute_tool("read_profile", {}, ctx)
    assert "75.0%" in result
    assert "2048.0 MB" in result


def test_readFile(tmp_path):
    (tmp_path / "test.py").write_text("print('hello')")
    ctx = _makeContext(tmp_path)
    result = execute_tool("read_file", {"path": "test.py"}, ctx)
    assert "print('hello')" in result


def test_readFileMissing(tmp_path):
    ctx = _makeContext(tmp_path)
    result = execute_tool("read_file", {"path": "nonexistent.py"}, ctx)
    assert "Error" in result


def test_listFiles(tmp_path):
    (tmp_path / "train.py").write_text("")
    (tmp_path / "model.py").write_text("")
    ctx = _makeContext(tmp_path)
    result = execute_tool("list_files", {}, ctx)
    assert "train.py" in result
    assert "model.py" in result


def test_searchCode(tmp_path):
    (tmp_path / "model.py").write_text("class MyModel(nn.Module):\n    pass\n")
    ctx = _makeContext(tmp_path)
    result = execute_tool("search_code", {"query": "nn.Module"}, ctx)
    assert "model.py" in result
    assert "MyModel" in result


def test_proposeConfig(tmp_path):
    ctx = _makeContext(tmp_path)
    result = execute_tool("propose_config", {
        "name": "Flash Attention",
        "description": "Enable flash attention",
        "optimization_type": "attention",
        "evidence": ["sdpa = 58% GPU time"],
        "estimated_speedup": 1.5,
        "risk_level": "low",
    }, ctx)
    parsed = json.loads(result)
    assert parsed["status"] == "accepted"
    assert parsed["name"] == "Flash Attention"


def test_unknownTool(tmp_path):
    ctx = _makeContext(tmp_path)
    result = execute_tool("fake_tool", {}, ctx)
    assert "Unknown tool" in result
