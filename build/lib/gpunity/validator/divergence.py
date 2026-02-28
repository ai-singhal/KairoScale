"""Divergence detection between control and optimized training runs.

Compares loss curves and gradient similarities to detect when an
optimized configuration produces numerically different results.
"""

from __future__ import annotations

import math
from typing import Any, Optional


def check_divergence(
    control_grads: list[dict[str, Any]],
    variant_grads: list[dict[str, Any]],
    threshold: float = 0.8,
    consecutive_failures: int = 3,
    loss_ratio_limit: float = 2.0,
) -> tuple[bool, Optional[int], str]:
    """Check if an optimized variant has diverged from the control run.

    Divergence is detected by:
    1. Loss ratio exceeding limit for consecutive checks.
    2. Loss becoming NaN or Inf.
    3. Gradient cosine similarity below threshold (if available).

    Args:
        control_grads: Control run checkpoints (step, loss, gradient_norm).
        variant_grads: Variant run checkpoints.
        threshold: Cosine similarity threshold (below = diverged).
        consecutive_failures: How many consecutive failures before flagging.
        loss_ratio_limit: Max allowed loss ratio (variant/control).

    Returns:
        Tuple of (diverged, divergence_step, reason).
    """
    if not control_grads or not variant_grads:
        return False, None, ""

    min_len = min(len(control_grads), len(variant_grads))
    failure_streak = 0

    for i in range(min_len):
        ctrl = control_grads[i]
        var = variant_grads[i]

        ctrl_loss = ctrl.get("loss", 0.0)
        var_loss = var.get("loss", 0.0)
        step = var.get("step", i)

        # Check for NaN/Inf
        if isinstance(var_loss, float) and (math.isnan(var_loss) or math.isinf(var_loss)):
            return True, step, f"Loss is {'NaN' if math.isnan(var_loss) else 'Inf'} at step {step}"

        # Check loss ratio
        if isinstance(ctrl_loss, (int, float)) and isinstance(var_loss, (int, float)):
            if ctrl_loss > 0 and var_loss > 0:
                ratio = var_loss / ctrl_loss
                if ratio > loss_ratio_limit:
                    failure_streak += 1
                    if failure_streak >= consecutive_failures:
                        return True, step, (
                            f"Loss ratio {ratio:.2f} exceeded limit {loss_ratio_limit} "
                            f"for {consecutive_failures} consecutive checks at step {step}"
                        )
                else:
                    failure_streak = 0

        # Check gradient cosine similarity if available
        ctrl_cos = ctrl.get("cosine_similarity")
        if ctrl_cos is not None and isinstance(ctrl_cos, (int, float)):
            if ctrl_cos < threshold:
                failure_streak += 1
                if failure_streak >= consecutive_failures:
                    return True, step, (
                        f"Gradient cosine similarity {ctrl_cos:.3f} below threshold "
                        f"{threshold} for {consecutive_failures} consecutive checks at step {step}"
                    )
            else:
                failure_streak = 0

    return False, None, ""


def compute_cosine_similarities(
    control_losses: list[float],
    variant_losses: list[float],
) -> list[float]:
    """Compute a loss-curve-based similarity metric between runs.

    Uses the ratio of losses as a proxy for divergence when gradient
    tensors are not available. Returns values between 0 and 1 where
    1 means identical loss trajectories.

    Args:
        control_losses: Loss values from the control run.
        variant_losses: Loss values from the variant run.

    Returns:
        List of similarity scores (one per step).
    """
    similarities = []
    min_len = min(len(control_losses), len(variant_losses))

    for i in range(min_len):
        c = control_losses[i]
        v = variant_losses[i]

        if math.isnan(c) or math.isnan(v) or math.isinf(c) or math.isinf(v):
            similarities.append(0.0)
            continue

        if c == 0 and v == 0:
            similarities.append(1.0)
            continue

        if c == 0 or v == 0:
            similarities.append(0.0)
            continue

        # Similarity = 1 - |log(v/c)| / log(loss_ratio_limit)
        # Clamp to [0, 1]
        ratio = v / c
        if ratio <= 0:
            similarities.append(0.0)
            continue

        log_ratio = abs(math.log(ratio))
        sim = max(0.0, 1.0 - log_ratio / math.log(2.0))
        similarities.append(min(1.0, sim))

    return similarities


def compare_logit_signatures(
    control_grads: list[dict[str, Any]],
    variant_grads: list[dict[str, Any]],
    tolerance: float,
) -> tuple[int, Optional[float], Optional[float], Optional[int]]:
    """Compare per-step logit signatures between control and variant runs.

    The wrapper stores lightweight signatures (mean/std/min/max/l2/sample)
    instead of raw logits. This function computes absolute deltas across shared
    keys and returns aggregate differences.

    Returns:
        (checks_compared, max_abs_diff, mean_abs_diff, first_failing_step)
    """
    comparable_diffs: list[tuple[int, float]] = []
    min_len = min(len(control_grads), len(variant_grads))

    for i in range(min_len):
        ctrl_sig = control_grads[i].get("logit_signature")
        var_sig = variant_grads[i].get("logit_signature")
        if not isinstance(ctrl_sig, dict) or not isinstance(var_sig, dict):
            continue

        diffs: list[float] = []
        for key in ("mean", "std", "min", "max", "l2"):
            c_val = ctrl_sig.get(key)
            v_val = var_sig.get(key)
            if isinstance(c_val, (int, float)) and isinstance(v_val, (int, float)):
                diffs.append(abs(float(c_val) - float(v_val)))

        c_sample = ctrl_sig.get("sample")
        v_sample = var_sig.get("sample")
        if isinstance(c_sample, list) and isinstance(v_sample, list):
            for c_val, v_val in zip(c_sample, v_sample):
                if isinstance(c_val, (int, float)) and isinstance(v_val, (int, float)):
                    diffs.append(abs(float(c_val) - float(v_val)))

        if diffs:
            comparable_diffs.append((i, max(diffs)))

    if not comparable_diffs:
        return 0, None, None, None

    max_diff = max(d for _, d in comparable_diffs)
    mean_diff = sum(d for _, d in comparable_diffs) / len(comparable_diffs)
    first_failing_step = next((step for step, d in comparable_diffs if d > tolerance), None)
    return len(comparable_diffs), max_diff, mean_diff, first_failing_step
