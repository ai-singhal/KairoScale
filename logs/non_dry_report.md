# GPUnity Optimization Report

> **Repo**: `/Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/tests/fixtures/nano_gpt` | **GPU**: a100-80gb | **Workload**: train | **Objective**: balanced | **Date**: 2026-02-28 16:41 | **Mode**: local


## Executive Summary

- **Best overall config ID**: `opt-001`
- **Best config**: "Swap optimizer + Triton fused kernels" -- 1.0x speedup, +0% memory
- 1/3 configs validated successfully
- 2/3 configs executed without runtime failure
- 1 config(s) showed divergence
- Control run cost: $0.0001

## Hardware Context

- GPU: NVIDIA A100 80GB
- Detection source: declared-gpu (medium confidence)
- Compute capability: 8.0
- VRAM: 81920 MB
- Feature flags: compile=True, cuda_graphs=True, bf16=True, tf32=True
- Note: Runtime CUDA inspection unavailable; used gpu_type priors.

## Profile Analysis

### Key Findings

- GPU Utilization: 0.0%
- Peak Memory: 0.0 MB
- Forward/Backward split: 30% / 70%
- Backward/Forward ratio: 2.3x
- DataLoader bottleneck: No
- DataLoader stall: 0.20 ms/step
- Training loop detection: heuristic

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-001 | Swap optimizer + Triton fused kernels | kernel_fusion | 1.1x | medium | Backward/forward ratio is 2.33x; High backward dominance can benefit from optimizer/kernel improvements |

### opt-001: Swap optimizer + Triton fused kernels

**Type**: kernel_fusion | **Risk**: medium | **Estimated speedup**: 1.1x | **Memory delta**: -5%

Evaluate SGD/RMSProp/AdamW/Adafactor/MUON variants plus Triton-fused optimizer kernels where available.

**Evidence**:
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements

**Config overrides**: `{'optimizer_strategy': 'fused_triton_search', 'optimizer_candidates': ['sgd', 'rmsprop', 'adamw', 'adafactor', 'muon']}`

**Config JSON**: `/Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/logs/gpunity_configs/opt-001.json`
**Apply command**: `gpunity apply /Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/logs/gpunity_configs/opt-001.json --repo /Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/tests/fixtures/nano_gpt`

## Validation Results

### Summary

| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |
|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|
| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |
| torch.compile(max-autotune + CUDA Graphs) | FAILED | - | - | - | - | - | - | - |
| torch.compile(max-autotune-no-cudagraphs) | 0.01x | -99.2% | +0.0% | +12034.6% | N/A | N/A | YES | 0.466571 |
| Swap optimizer + Triton fused kernels | 1.01x | +1.3% | +0.0% | -1.2% | 0.004 | N/A | No | 0.000000 |


**Control run**: 26 steps, 0.1s, 4.2 ms/step, peak memory 0.0 MB

### Loss Curves

| Step | Control | torch.compile(max-au | Swap optimizer + Tri |
| --- | --- | --- | --- |
| 0 | 5.6456 | 5.6392 | 5.6456 |
| 2 | 5.6727 | 5.6623 | 5.6727 |
| 4 | 5.6847 | 5.6959 | 5.6847 |
| 6 | 5.7070 | 5.7149 | 5.7070 |
| 8 | 5.7072 | 5.7172 | 5.7072 |
| 10 | 5.6713 | 5.6773 | 5.6713 |
| 12 | 5.6982 | 5.6908 | 5.6982 |
| 14 | 5.6810 | 5.6961 | 5.6810 |
| 16 | 5.6122 | 5.6134 | 5.6122 |
| 18 | 5.6034 | 5.6076 | 5.6034 |
| 20 | 5.6513 | 5.6592 | 5.6513 |
| 22 | 5.6397 | 5.6212 | 5.6397 |
| 24 | 5.6042 | 5.6088 | 5.6042 |

### Gradient Similarity

| Step | torch.compile(max-au | Swap optimizer + Tri | Threshold |
| --- | --- | --- | --- |
| 0 | 0.998 | 1.000 | 0.80 |
| 2 | 0.997 | 1.000 | 0.80 |
| 4 | 0.997 | 1.000 | 0.80 |
| 6 | 0.998 | 1.000 | 0.80 |
| 8 | 0.997 | 1.000 | 0.80 |
| 10 | 0.998 | 1.000 | 0.80 |
| 12 | 0.998 | 1.000 | 0.80 |
| 14 | 0.996 | 1.000 | 0.80 |
| 16 | 1.000 | 1.000 | 0.80 |
| 18 | 0.999 | 1.000 | 0.80 |
| 20 | 0.998 | 1.000 | 0.80 |
| 22 | 0.995 | 1.000 | 0.80 |
| 24 | 0.999 | 1.000 | 0.80 |

### Correctness (Logit Signatures)

- **torch.compile(max-autotune-no-cudagraphs)**: checks=26, max_abs_diff=0.466571, mean_abs_diff=0.291837 (exceeded tolerance)
- **Swap optimizer + Triton fused kernels**: checks=26, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
### Divergence Flags

- **torch.compile(max-autotune-no-cudagraphs)**: Diverged at step 0. Reason: Logit signature delta 0.466571 exceeded tolerance 0.001000

## Evidence Tracebacks

### Evidence Graph

- `Native baseline candidate required by benchmark ladder.` -> `B1` (supports)
- `Detected GPU count: 1` -> `B1` (supports)
- `Fallback mode: default` -> `B1` (supports)
- `Backward/forward ratio is 2.33x` -> `opt-001` (supports)
- `High backward dominance can benefit from optimizer/kernel improvements` -> `opt-001` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-001` (supports)

### Swap optimizer + Triton fused kernels

**Baseline comparison**
- vs control speedup: 1.01x
- vs control cost delta: -1.2%
- vs control throughput delta: +1.3%

**Profiler traceback**
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 26
- Logit max abs diff: 0.000000
- Objective score: 0.004

**Code traceback**
- No code patch required (runtime/config optimization).

## Native Baseline Ladder

| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |
|----------|----------|---------|--------------------|--------------|--------|-----------|
| B0 (Control (eager, no optimization)) | YES | YES | - | - | - | - |
| B1 (torch.compile(max-autotune-no-cudagraphs)) | YES | No | 0.01x | -99.2% | +12034.6% | - |
| B2 (torch.compile(max-autotune + CUDA Graphs)) | No | No | - | - | - | - |

- `B2` skipped: Reserved for multi-GPU cluster runs

## Recommendations

1. **Consider**: Swap optimizer + Triton fused kernels -- 1.0x speedup, +0% memory
2. **Skip**: torch.compile(max-autotune + CUDA Graphs) -- Failed to run
3. **Skip**: torch.compile(max-autotune-no-cudagraphs) -- Diverged: Logit signature delta 0.466571 exceeded tolerance 0.001000

## Appendix


### Raw Config Data

**opt-001: Swap optimizer + Triton fused kernels**

```json
{
  "id": "opt-001",
  "name": "Swap optimizer + Triton fused kernels",
  "description": "Evaluate SGD/RMSProp/AdamW/Adafactor/MUON variants plus Triton-fused optimizer kernels where available.",
  "optimization_type": "kernel_fusion",
  "evidence": [
    "Backward/forward ratio is 2.33x",
    "High backward dominance can benefit from optimizer/kernel improvements"
  ],
  "code_changes": {},
  "config_overrides": {
    "optimizer_strategy": "fused_triton_search",
    "optimizer_candidates": [
      "sgd",
      "rmsprop",
      "adamw",
      "adafactor",
      "muon"
    ]
  },
  "estimated_speedup": 1.12,
  "estimated_memory_delta": -0.05,
  "risk_level": "medium",
  "dependencies": [],
  "is_native_baseline": false,
  "baseline_id": null,
  "eligible": true,
  "ineligible_reason": null,
  "heuristic_rationale": []
}
```


---
*Generated by GPUnity v0.1.0*