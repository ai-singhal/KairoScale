"""Tests for gpunity.validator.divergence."""

import math

from gpunity.validator.divergence import check_divergence, compute_cosine_similarities


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
