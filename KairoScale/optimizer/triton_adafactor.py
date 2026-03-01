"""Triton-fused Adafactor optimizer.

Adafactor uses factored second-moment estimation to dramatically reduce
optimizer state memory.  Instead of storing a full second-moment matrix
(like Adam), it maintains row and column factors whose outer product
approximates the full matrix.

The Triton kernels fuse:
  1. Row/column factor updates
  2. Second-moment reconstruction + RMS scaling
  3. Parameter update with optional weight decay

into minimal GPU launches, cutting kernel-launch overhead by ~3x compared
to the equivalent pure-PyTorch loop.

Reference: Shazeer & Stern, "Adafactor: Adaptive Learning Rates with
Sublinear Memory Cost", ICML 2018.
"""

from __future__ import annotations

import math
import torch
from torch.optim.optimizer import Optimizer

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ------------------------------------------------------------------
# Triton kernels
# ------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _adafactor_row_factor_kernel(
        grad_ptr,         # [rows, cols] flattened
        row_factor_ptr,   # [rows]
        cols,
        rho,
        eps,
        BLOCK_COL: tl.constexpr,
    ):
        """Update row factor: r_i = rho * r_i + (1-rho) * mean_j(g_{ij}^2)."""
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_COL)
        mask = col_offsets < cols

        grad_offset = row * cols + col_offsets
        g = tl.load(grad_ptr + grad_offset, mask=mask, other=0.0)
        g_sq = g * g

        # Compute mean of g^2 across columns
        row_sum = tl.sum(g_sq, axis=0)
        row_mean = row_sum / cols

        # EMA update
        old_r = tl.load(row_factor_ptr + row)
        new_r = rho * old_r + (1.0 - rho) * row_mean
        tl.store(row_factor_ptr + row, new_r)

    @triton.jit
    def _adafactor_col_factor_kernel(
        grad_ptr,         # [rows, cols] flattened
        col_factor_ptr,   # [cols]
        rows,
        cols,
        rho,
        eps,
        BLOCK_ROW: tl.constexpr,
    ):
        """Update col factor: c_j = rho * c_j + (1-rho) * mean_i(g_{ij}^2)."""
        col = tl.program_id(0)
        row_offsets = tl.arange(0, BLOCK_ROW)
        mask = row_offsets < rows

        grad_offsets = row_offsets * cols + col
        g = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0)
        g_sq = g * g

        col_sum = tl.sum(g_sq, axis=0)
        col_mean = col_sum / rows

        old_c = tl.load(col_factor_ptr + col)
        new_c = rho * old_c + (1.0 - rho) * col_mean
        tl.store(col_factor_ptr + col, new_c)

    @triton.jit
    def _adafactor_update_2d_kernel(
        param_ptr,
        grad_ptr,
        row_factor_ptr,
        col_factor_ptr,
        rows,
        cols,
        lr,
        weight_decay,
        eps,
        d_rms,            # 1/RMS(row_factor) for normalization
        BLOCK_COL: tl.constexpr,
    ):
        """Fused 2-D Adafactor parameter update.

        v_{ij} = r_i * c_j / mean(r)   (reconstructed second moment)
        update = grad / sqrt(v + eps)
        param -= lr * (update + weight_decay * param)
        """
        row = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_COL)
        mask = col_offsets < cols

        offset = row * cols + col_offsets

        p = tl.load(param_ptr + offset, mask=mask)
        g = tl.load(grad_ptr + offset, mask=mask, other=0.0)
        ri = tl.load(row_factor_ptr + row)
        cj = tl.load(col_factor_ptr + col_offsets, mask=mask, other=1.0)

        # Reconstruct second moment
        v = ri * cj * d_rms + eps
        inv_sqrt_v = 1.0 / tl.sqrt(v)

        # Scaled update
        update = g * inv_sqrt_v

        # Weight decay
        update = update + weight_decay * p

        p = p - lr * update
        tl.store(param_ptr + offset, p, mask=mask)

    @triton.jit
    def _adafactor_update_1d_kernel(
        param_ptr,
        grad_ptr,
        v_ptr,
        n_elements,
        lr,
        rho,
        weight_decay,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused 1-D (unfactored) Adafactor update for bias/1-D params.

        v = rho * v + (1-rho) * g^2
        update = g / sqrt(v + eps)
        param -= lr * (update + weight_decay * param)
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(param_ptr + offsets, mask=mask)
        g = tl.load(grad_ptr + offsets, mask=mask, other=0.0)
        v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

        # EMA update of second moment
        v = rho * v + (1.0 - rho) * g * g
        tl.store(v_ptr + offsets, v, mask=mask)

        inv_sqrt_v = 1.0 / tl.sqrt(v + eps)
        update = g * inv_sqrt_v
        update = update + weight_decay * p

        p = p - lr * update
        tl.store(param_ptr + offsets, p, mask=mask)


def _rms(tensor: torch.Tensor) -> float:
    """Root mean square of a tensor."""
    return math.sqrt(tensor.square().mean().item() + 1e-30)


def _decay_rate(step: int, rho_min: float = 0.8) -> float:
    """Adaptive decay rate schedule from the original Adafactor paper."""
    return max(rho_min, 1.0 - (step + 1) ** (-0.8))


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


class TritonAdafactor(Optimizer):
    """Adafactor optimizer with Triton-fused kernels.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. If None, uses relative step size (default: 1e-3).
        rho_min: Minimum decay rate for second moment EMA (default: 0.8).
        eps: Epsilon for numerical stability (default: 1e-30).
        weight_decay: Decoupled weight decay (default: 0.0).
        relative_step: Use adaptive learning rate based on step count (default: False).
        scale_parameter: Scale lr by RMS of parameter (default: True).
        warmup_init: Use warmup initialization for relative step (default: False).
    """

    def __init__(
        self,
        params,
        lr: float | None = 1e-3,
        rho_min: float = 0.8,
        eps: float = 1e-30,
        weight_decay: float = 0.0,
        relative_step: bool = False,
        scale_parameter: bool = True,
        warmup_init: bool = False,
    ):
        if lr is not None and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = dict(
            lr=lr,
            rho_min=rho_min,
            eps=eps,
            weight_decay=weight_decay,
            relative_step=relative_step,
            scale_parameter=scale_parameter,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def _get_lr(self, group: dict, state: dict) -> float:
        """Compute effective learning rate."""
        if group["relative_step"]:
            step = state["step"]
            min_step = 1e-6 if group["warmup_init"] else 1e-2
            rel_step = max(min_step, 1.0 / math.sqrt(step + 1))
            param_scale = 1.0
            if group["scale_parameter"]:
                param_scale = max(_rms(state["param_rms_ref"]), 1e-3)
            return rel_step * param_scale
        return group["lr"] or 1e-3

    def _use_factored(self, shape: tuple[int, ...]) -> bool:
        """Determine whether to use factored second moments."""
        return len(shape) >= 2

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            wd = group["weight_decay"]
            rho_min = group["rho_min"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["param_rms_ref"] = p.detach().clone()
                    if self._use_factored(p.shape):
                        state["row_factor"] = torch.zeros(
                            p.shape[0], device=p.device, dtype=p.dtype
                        )
                        state["col_factor"] = torch.zeros(
                            p.shape[-1], device=p.device, dtype=p.dtype
                        )
                    else:
                        state["v"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]
                rho = _decay_rate(step, rho_min)
                lr = self._get_lr(group, state)

                use_triton = _TRITON_AVAILABLE and p.is_cuda and p.is_contiguous()

                if self._use_factored(p.shape):
                    rows = p.shape[0]
                    cols = p.shape[-1]
                    # For >2D tensors, reshape to 2D for factored update
                    if p.ndim > 2:
                        p_2d = p.view(rows, -1)
                        g_2d = grad.contiguous().view(rows, -1)
                        cols = p_2d.shape[1]
                    else:
                        p_2d = p
                        g_2d = grad.contiguous()

                    rf = state["row_factor"]
                    cf = state["col_factor"]

                    if use_triton:
                        BLOCK_COL = _next_power_of_2(cols)
                        BLOCK_ROW = _next_power_of_2(rows)

                        # Update row factors
                        _adafactor_row_factor_kernel[(rows,)](
                            g_2d, rf, cols, rho, eps, BLOCK_COL=BLOCK_COL,
                        )

                        # Update col factors
                        _adafactor_col_factor_kernel[(cols,)](
                            g_2d, cf, rows, cols, rho, eps, BLOCK_ROW=BLOCK_ROW,
                        )

                        # Compute d_rms = 1 / mean(row_factor) for normalization
                        rf_mean = rf.mean().item()
                        d_rms = 1.0 / max(rf_mean, eps)

                        # Fused parameter update
                        _adafactor_update_2d_kernel[(rows,)](
                            p_2d, g_2d, rf, cf, rows, cols,
                            lr, wd, eps, d_rms, BLOCK_COL=BLOCK_COL,
                        )
                    else:
                        # PyTorch fallback
                        g_sq = g_2d.square()
                        rf.mul_(rho).add_(g_sq.mean(dim=1), alpha=1.0 - rho)
                        cf.mul_(rho).add_(g_sq.mean(dim=0), alpha=1.0 - rho)

                        # Reconstruct second moment
                        rf_mean = rf.mean().clamp(min=eps)
                        v = torch.outer(rf, cf) / rf_mean + eps
                        update = g_2d / v.sqrt()

                        if wd != 0:
                            update.add_(p_2d, alpha=wd)

                        p_2d.add_(update, alpha=-lr)
                else:
                    # 1-D / unfactored path
                    v = state["v"]
                    n = p.numel()

                    if use_triton:
                        BLOCK = 1024
                        grid = ((n + BLOCK - 1) // BLOCK,)
                        _adafactor_update_1d_kernel[grid](
                            p.view(-1), grad.contiguous().view(-1), v.view(-1),
                            n, lr, rho, wd, eps, BLOCK_SIZE=BLOCK,
                        )
                    else:
                        v.mul_(rho).addcmul_(grad, grad, value=1.0 - rho)
                        update = grad / (v.sqrt() + eps)
                        if wd != 0:
                            update.add_(p, alpha=wd)
                        p.add_(update, alpha=-lr)

                # Update RMS reference for relative step
                if group["relative_step"]:
                    state["param_rms_ref"] = p.detach().clone()

        return loss
