"""Tests for KairoScale.utils modules."""

from pathlib import Path

from KairoScale.utils.cost import GPU_COSTS_PER_HOUR, estimate_cost
from KairoScale.utils.repo import detect_dependencies, scan_repo


class TestScanRepo:
    def test_findsFiles(self, tmp_path):
        (tmp_path / "train.py").write_text("print('train')")
        (tmp_path / "model.py").write_text("print('model')")
        result = scan_repo(tmp_path)
        assert "train.py" in result["python_files"]
        assert "model.py" in result["python_files"]
        assert "train.py" in result["entry_candidates"]

    def test_detectsRequirements(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch\n")
        result = scan_repo(tmp_path)
        assert result["has_requirements"] is True

    def test_ignoresHiddenDirs(self, tmp_path):
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("test")
        (tmp_path / "main.py").write_text("")
        result = scan_repo(tmp_path)
        assert all(".git" not in f for f in result["files"])


class TestDetectDependencies:
    def test_fromRequirementsTxt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("torch>=2.0\nnumpy\n# comment\n")
        deps = detect_dependencies(tmp_path)
        assert "torch>=2.0" in deps
        assert "numpy" in deps
        assert len(deps) == 2

    def test_emptyRepo(self, tmp_path):
        deps = detect_dependencies(tmp_path)
        assert deps == []


class TestEstimateCost:
    def test_knownGpu(self):
        cost = estimate_cost("a100-80gb", 3600)
        assert cost == GPU_COSTS_PER_HOUR["a100-80gb"]

    def test_fractionalHour(self):
        cost = estimate_cost("a100-80gb", 1800)
        assert cost == GPU_COSTS_PER_HOUR["a100-80gb"] / 2
