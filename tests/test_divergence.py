"""Tests for KairoScale.validator.divergence."""

import math

from KairoScale.validator.divergence import (
    check_divergence,
    compare_logit_signatures,
    compute_cosine_similarities,
)


class TestCheckDivergence:
    def test_noDivergence(self):
        control = [{"step": i, "loss": 1.0} for i in range(10)]
        variant = [{"step": i, "loss": 1.0} for i in range(10)]
        diverged, step, reason = check_divergence(control, variant)
        assert diverged is False
        assert step is None

    def test_nanDivergence(self):
        control = [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.9}]
        variant = [{"step": 0, "loss": 1.0}, {"step": 1, "loss": float("nan")}]
        diverged, step, reason = check_divergence(control, variant)
        assert diverged is True
        assert "NaN" in reason

    def test_infDivergence(self):
        control = [{"step": 0, "loss": 1.0}]
        variant = [{"step": 0, "loss": float("inf")}]
        diverged, step, reason = check_divergence(control, variant)
        assert diverged is True
        assert "Inf" in reason

    def test_lossRatioDivergence(self):
        control = [{"step": i, "loss": 1.0} for i in range(5)]
        variant = [{"step": i, "loss": 5.0} for i in range(5)]
        diverged, step, reason = check_divergence(
            control, variant, consecutive_failures=3, loss_ratio_limit=2.0,
        )
        assert diverged is True
        assert "ratio" in reason.lower()

    def test_emptyInputs(self):
        diverged, step, reason = check_divergence([], [])
        assert diverged is False

    def test_singleFailureNotEnough(self):
        control = [{"step": 0, "loss": 1.0}, {"step": 1, "loss": 1.0}]
        variant = [{"step": 0, "loss": 5.0}, {"step": 1, "loss": 1.0}]
        diverged, step, reason = check_divergence(
            control, variant, consecutive_failures=3,
        )
        assert diverged is False


class TestComputeCosineSimilarities:
    def test_identicalLosses(self):
        sims = compute_cosine_similarities([1.0, 0.9, 0.8], [1.0, 0.9, 0.8])
        assert all(s == 1.0 for s in sims)

    def test_divergentLosses(self):
        sims = compute_cosine_similarities([1.0, 1.0], [10.0, 10.0])
        assert all(s < 1.0 for s in sims)

    def test_nanHandling(self):
        sims = compute_cosine_similarities([1.0, float("nan")], [1.0, 1.0])
        assert sims[0] == 1.0
        assert sims[1] == 0.0

    def test_zeroHandling(self):
        sims = compute_cosine_similarities([0.0, 0.0], [0.0, 0.0])
        assert sims == [1.0, 1.0]

    def test_differentLengths(self):
        sims = compute_cosine_similarities([1.0, 0.9, 0.8], [1.0, 0.9])
        assert len(sims) == 2


class TestCompareLogitSignatures:
    def test_withinTolerance(self):
        control = [{"logit_signature": {"mean": 1.0, "std": 2.0, "sample": [0.1, 0.2]}}]
        variant = [{"logit_signature": {"mean": 1.0001, "std": 2.0001, "sample": [0.1, 0.2]}}]
        checks, max_diff, mean_diff, failing_step = compare_logit_signatures(
            control, variant, tolerance=1e-2
        )
        assert checks == 1
        assert max_diff is not None and max_diff < 1e-2
        assert mean_diff is not None
        assert failing_step is None

    def test_exceedsTolerance(self):
        control = [{"logit_signature": {"mean": 1.0, "std": 2.0, "sample": [0.1, 0.2]}}]
        variant = [{"logit_signature": {"mean": 1.5, "std": 2.0, "sample": [0.1, 0.2]}}]
        checks, max_diff, mean_diff, failing_step = compare_logit_signatures(
            control, variant, tolerance=1e-2
        )
        assert checks == 1
        assert max_diff is not None and max_diff > 1e-2
        assert failing_step == 0
