# GPUnity Optimization Report

> **Repo**: `/Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet` | **GPU**: a100-80gb | **Workload**: train | **Objective**: balanced | **Date**: 2026-02-28 17:18 | **Mode**: modal


## Executive Summary

- **Best overall config ID**: `opt-002`
- **Best config**: "Enable selective gradient checkpointing" -- 0.9x speedup, +0% memory
- 2/4 configs validated successfully
- 3/4 configs executed without runtime failure
- 1 config(s) showed divergence
- Control run cost: $0.0004

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
- Peak Memory: 11.1 MB
- Forward/Backward split: 30% / 70%
- Backward/Forward ratio: 2.3x
- DataLoader bottleneck: No
- DataLoader stall: 12.92 ms/step
- Training loop detection: none

> **Warning**: Training loop could not be automatically detected. Consider using `--train-function <name>` for more accurate profiling.

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-001 | Swap optimizer + Triton fused kernels | kernel_fusion | 1.1x | medium | Backward/forward ratio is 2.33x; High backward dominance can benefit from optimizer/kernel improvements |
| opt-002 | Enable selective gradient checkpointing | memory | 0.9x | medium | Peak memory reached 11.1 MB; Activation memory pressure can limit larger batch sizes |

### opt-001: Swap optimizer + Triton fused kernels

**Type**: kernel_fusion | **Risk**: medium | **Estimated speedup**: 1.1x | **Memory delta**: -5%

Evaluate SGD/RMSProp/AdamW/Adafactor/MUON variants plus Triton-fused optimizer kernels where available.

**Evidence**:
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements

**Config overrides**: `{'optimizer_strategy': 'fused_triton_search', 'optimizer_candidates': ['sgd', 'rmsprop', 'adamw', 'adafactor', 'muon']}`

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

### opt-002: Enable selective gradient checkpointing

**Type**: memory | **Risk**: medium | **Estimated speedup**: 0.9x | **Memory delta**: -35%

Checkpoint high-memory blocks to trade extra compute for lower peak activation memory.

**Evidence**:
- Peak memory reached 11.1 MB
- Activation memory pressure can limit larger batch sizes

**Config overrides**: `{'gradient_checkpointing': True}`

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

## Validation Results

### Summary

| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |
|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|
| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |
| torch.compile(max-autotune + CUDA Graphs) | FAILED | - | - | - | - | - | - | - |
| torch.compile(max-autotune-no-cudagraphs) | 0.01x | -99.2% | +393.7% | +12538.8% | N/A | N/A | YES | 0.016756 |
| Swap optimizer + Triton fused kernels | 0.75x | -24.9% | +0.0% | +33.1% | -0.264 | N/A | No | 0.000000 |
| Enable selective gradient checkpointing | 0.91x | -8.6% | +0.0% | +9.5% | -0.092 | N/A | No | 0.000000 |


**Control run**: 4 steps, 0.4s, 137.0 ms/step, peak memory 11.1 MB

### Loss Curves

| Step | Control | torch.compile(max-au | Swap optimizer + Tri | Enable selective gra |
| --- | --- | --- | --- | --- |
| 0 | 0.7051 | 0.7051 | 0.7051 | 0.7051 |
| 1 | 0.7081 | 0.7080 | 0.7081 | 0.7081 |
| 2 | 0.7008 | 0.7007 | 0.7008 | 0.7008 |
| 3 | 0.7055 | 0.7056 | 0.7055 | 0.7055 |

### Gradient Similarity

| Step | torch.compile(max-au | Swap optimizer + Tri | Enable selective gra | Threshold |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 1.000 | 1.000 | 0.80 |
| 1 | 1.000 | 1.000 | 1.000 | 0.80 |
| 2 | 1.000 | 1.000 | 1.000 | 0.80 |
| 3 | 1.000 | 1.000 | 1.000 | 0.80 |

### Correctness (Logit Signatures)

- **torch.compile(max-autotune-no-cudagraphs)**: checks=4, max_abs_diff=0.016756, mean_abs_diff=0.008846 (exceeded tolerance)
- **Swap optimizer + Triton fused kernels**: checks=4, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
- **Enable selective gradient checkpointing**: checks=4, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
### Divergence Flags

- **torch.compile(max-autotune-no-cudagraphs)**: Diverged at step 1. Reason: Logit signature delta 0.016756 exceeded tolerance 0.001000

## Evidence Tracebacks

### Evidence Graph

- `Native baseline candidate required by benchmark ladder.` -> `B1` (supports)
- `Detected GPU count: 1` -> `B1` (supports)
- `Fallback mode: default` -> `B1` (supports)
- `Backward/forward ratio is 2.33x` -> `opt-001` (supports)
- `High backward dominance can benefit from optimizer/kernel improvements` -> `opt-001` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-001` (supports)
- `Peak memory reached 11.1 MB` -> `opt-002` (supports)
- `Activation memory pressure can limit larger batch sizes` -> `opt-002` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-002` (supports)

### Enable selective gradient checkpointing

**Baseline comparison**
- vs control speedup: 0.91x
- vs control cost delta: +9.5%
- vs control throughput delta: -8.6%

**Profiler traceback**
- Peak memory reached 11.1 MB
- Activation memory pressure can limit larger batch sizes
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 4
- Logit max abs diff: 0.000000
- Objective score: -0.092

**Code traceback**
- No code patch required (runtime/config optimization).
### Swap optimizer + Triton fused kernels

**Baseline comparison**
- vs control speedup: 0.75x
- vs control cost delta: +33.1%
- vs control throughput delta: -24.9%

**Profiler traceback**
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 4
- Logit max abs diff: 0.000000
- Objective score: -0.264

**Code traceback**
- No code patch required (runtime/config optimization).

## Native Baseline Ladder

| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |
|----------|----------|---------|--------------------|--------------|--------|-----------|
| B0 (Control (eager, no optimization)) | YES | YES | - | - | - | - |
| B1 (torch.compile(max-autotune-no-cudagraphs)) | YES | No | 0.01x | -99.2% | +12538.8% | - |
| B2 (torch.compile(max-autotune + CUDA Graphs)) | No | No | - | - | - | - |

- `B2` skipped: Reserved for multi-GPU cluster runs

## Recommendations

1. **Skip**: torch.compile(max-autotune + CUDA Graphs) -- Failed to run
2. **Skip**: torch.compile(max-autotune-no-cudagraphs) -- Diverged: Logit signature delta 0.016756 exceeded tolerance 0.001000
3. **Skip**: Swap optimizer + Triton fused kernels -- Slower than baseline
4. **Skip**: Enable selective gradient checkpointing -- Slower than baseline

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

**opt-002: Enable selective gradient checkpointing**

```json
{
  "id": "opt-002",
  "name": "Enable selective gradient checkpointing",
  "description": "Checkpoint high-memory blocks to trade extra compute for lower peak activation memory.",
  "optimization_type": "memory",
  "evidence": [
    "Peak memory reached 11.1 MB",
    "Activation memory pressure can limit larger batch sizes"
  ],
  "code_changes": {},
  "config_overrides": {
    "gradient_checkpointing": true
  },
  "estimated_speedup": 0.95,
  "estimated_memory_delta": -0.35,
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