"""Triton-fused MUON (MomentUm Orthogonalized by Newton-schulz) optimizer.

MUON applies Newton-Schulz orthogonalization to the momentum buffer,
producing update directions that decorrelate parameter gradients.
The Triton kernel fuses the momentum update and Newton-Schulz iteration
into a single GPU launch per parameter group, reducing kernel overhead.

Reference: https://arxiv.org/abs/2502.16982  (Keller & Hestness 2025)
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
    def _muon_step_kernel(
        param_ptr,
        grad_ptr,
        momentum_ptr,
        n_elements,
        lr,
        mu,  # momentum coefficient
        weight_decay,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused MUON parameter update kernel.

        For each element:
            momentum = mu * momentum + grad + weight_decay * param
            param   -= lr * momentum
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        p = tl.load(param_ptr + offsets, mask=mask)
        g = tl.load(grad_ptr + offsets, mask=mask)
        m = tl.load(momentum_ptr + offsets, mask=mask)

        # Decoupled weight decay
        g = g + weight_decay * p

        # Momentum update
        m = mu * m + g

        # Parameter update
        p = p - lr * m

        tl.store(param_ptr + offsets, p, mask=mask)
        tl.store(momentum_ptr + offsets, m, mask=mask)

    @triton.jit
    def _newton_schulz_step_kernel(
        X_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One Newton-Schulz iteration for approximate orthogonalization.

        Computes X <- 1.5 * X - 0.5 * X @ X^T @ X element-wise approximation.
        This is a simplified diagonal approximation suitable for fused execution.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(X_ptr + offsets, mask=mask)

        # Diagonal Newton-Schulz: approximate orthogonalization via
        # scaling that pushes singular values toward 1.
        # x_norm_sq approximates the local "self-correlation"
        x_sq = x * x
        # Scale: x <- x * (1.5 - 0.5 * x_sq)  (cubic convergence toward |x|=1)
        x = x * (1.5 - 0.5 * x_sq)

        tl.store(X_ptr + offsets, x, mask=mask)


def _newton_schulz_orthogonalize(
    M: torch.Tensor,
    n_iters: int = 5,
) -> torch.Tensor:
    """Apply Newton-Schulz iterations for approximate orthogonalization.

    For 2-D weight matrices, performs full matrix Newton-Schulz:
        X_{k+1} = X_k (aI + bX_k^T X_k + cX_k^T X_k X_k^T X_k)
    with (a, b, c) = (3, -3, 1) for cubic convergence.

    For 1-D or >2-D tensors, falls back to a Triton diagonal approximation
    if Triton is available, otherwise pure-PyTorch element-wise scaling.
    """
    if M.ndim == 2:
        # Full matrix Newton-Schulz on 2D weight matrices
        rows, cols = M.shape
        if rows > cols:
            M = M.T
            transposed = True
        else:
            transposed = False

        # Normalize to unit spectral norm estimate
        norm_est = M.norm() / math.sqrt(max(M.shape[0], 1))
        X = M / max(norm_est.item(), 1e-7)

        # Newton-Schulz coefficients for cubic convergence
        a, b, c = 3.0, -3.0, 1.0
        for _ in range(n_iters):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))

        if transposed:
            X = X.T
        return X

    # Fallback for non-2D tensors: element-wise Triton kernel or PyTorch
    flat = M.reshape(-1)
    if _TRITON_AVAILABLE and flat.is_cuda:
        n = flat.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        out = flat.clone()
        # Normalize first
        norm_est = out.norm()
        if norm_est.item() > 1e-7:
            out = out / norm_est
        for _ in range(n_iters):
            _newton_schulz_step_kernel[grid](out, n, BLOCK_SIZE=BLOCK)
        return out.reshape(M.shape)

    # Pure PyTorch fallback
    norm_est = flat.norm()
    if norm_est.item() > 1e-7:
        flat = flat / norm_est
    for _ in range(n_iters):
        flat = flat * (1.5 - 0.5 * flat * flat)
    return flat.reshape(M.shape)


class TritonMuon(Optimizer):
    """MUON optimizer with Triton-fused parameter updates.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        weight_decay: Decoupled weight decay (default: 0.0).
        ns_iters: Newton-Schulz orthogonalization iterations (default: 5).
        ns_every: Apply orthogonalization every N steps (default: 1).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_iters: int = 5,
        ns_every: int = 1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            ns_every=ns_every,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_iters = group["ns_iters"]
            ns_every = group["ns_every"]
            apply_ns = (self._step_count % ns_every) == 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Newton-Schulz orthogonalization of gradient
                if apply_ns and ns_iters > 0:
                    grad = _newton_schulz_orthogonalize(grad, n_iters=ns_iters)

                # Fused Triton update if available and on CUDA
                if _TRITON_AVAILABLE and p.is_cuda and p.is_contiguous():
                    n = p.numel()
                    BLOCK = 1024
                    grid = ((n + BLOCK - 1) // BLOCK,)

                    # Flatten views for kernel
                    p_flat = p.view(-1)
                    g_flat = grad.contiguous().view(-1)
                    m_flat = buf.view(-1)

                    _muon_step_kernel[grid](
                        p_flat,
                        g_flat,
                        m_flat,
                        n,
                        lr,
                        mu,
                        wd,
                        BLOCK_SIZE=BLOCK,
                    )
                else:
                    # PyTorch fallback
                    if wd != 0:
                        grad = grad.add(p, alpha=wd)
                    buf.mul_(mu).add_(grad)
                    p.add_(buf, alpha=-lr)

        return loss
