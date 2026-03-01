"""Tests for KairoScale.profiler modules."""

from pathlib import Path

from KairoScale.profiler.wrapper import create_profiling_wrapper, detect_training_loop


class TestDetectTrainingLoop:
    def test_detectsStandardLoop(self):
        code = """
def train():
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
"""
        result = detect_training_loop(code)
        assert result is not None
        assert result["function"] == "train"

    def test_detectsModuleLevelLoop(self):
        code = """
for epoch in range(10):
    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
"""
        result = detect_training_loop(code)
        assert result is not None
        assert result["function"] == "__main__"

    def test_returnsNoneWhenNoLoop(self):
        code = """
def inference():
    result = model(data)
    return result
"""
        result = detect_training_loop(code)
        assert result is None

    def test_returnsNoneOnSyntaxError(self):
        code = "def broken(:"
        result = detect_training_loop(code)
        assert result is None

    def test_needsBothBackwardAndStep(self):
        code = """
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    # no optimizer.step()
"""
        result = detect_training_loop(code)
        assert result is None


class TestCreateProfilingWrapper:
    def test_generatesValidPython(self):
        script = create_profiling_wrapper(
            repo_path=Path("/tmp/repo"),
            entry_point="train.py",
            train_function=None,
            warmup_steps=5,
            profile_steps=20,
        )
        # Should be valid Python
        compile(script, "<wrapper>", "exec")

    def test_containsStepCounts(self):
        script = create_profiling_wrapper(
            repo_path=Path("/tmp/repo"),
            entry_point="train.py",
            train_function=None,
            warmup_steps=3,
            profile_steps=10,
        )
        assert "WARMUP_STEPS = 3" in script
        assert "PROFILE_STEPS = 10" in script
        assert "TOTAL_STEPS = 13" in script

    def test_containsTrainFunction(self):
        script = create_profiling_wrapper(
            repo_path=Path("/tmp/repo"),
            entry_point="train.py",
            train_function="my_train",
            warmup_steps=5,
            profile_steps=20,
        )
        assert "'my_train'" in script

    def test_containsArtifactSaving(self):
        script = create_profiling_wrapper(
            repo_path=Path("/tmp/repo"),
            entry_point="train.py",
            train_function=None,
            warmup_steps=5,
            profile_steps=20,
        )
        assert "profile_result.json" in script
        assert "dataloader_stats.json" in script

    def test_containsFallbackEntryInvocation(self):
        script = create_profiling_wrapper(
            repo_path=Path("/tmp/repo"),
            entry_point="train.py",
            train_function=None,
            warmup_steps=5,
            profile_steps=20,
        )
        assert "_invoke_fallback_entry" in script
        assert "\"train\", \"main\", \"run\"" in script
