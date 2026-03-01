# GPUnity Optimization Report

> **Repo**: `/Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet` | **GPU**: a100-80gb | **Workload**: train | **Objective**: balanced | **Date**: 2026-02-28 19:27 | **Mode**: modal


## Executive Summary

- **Best overall config ID**: `opt-002`
- **Best config**: "Increase DataLoader Workers" -- 1.0x speedup, +0% memory
- 3/5 configs validated successfully
- 4/5 configs executed without runtime failure
- 1 config(s) showed divergence
- Control run cost: $0.0014

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
- Peak Memory: 31.1 MB
- Forward/Backward split: 30% / 70%
- Backward/Forward ratio: 2.3x
- DataLoader bottleneck: No
- DataLoader stall: 3.16 ms/step
- Training loop detection: none

> **Warning**: Training loop could not be automatically detected. Consider using `--train-function <name>` for more accurate profiling.

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-001 | Enable AMP for Mixed Precision Training | mixed_precision | 1.2x | low | Backward time: 17.67ms (70.0%); Backward/Forward ratio: 2.3x (+1 more) |
| opt-002 | Increase DataLoader Workers | data_loading | 1.1x | low | DataLoader throughput: 1265.8 samples/sec; DataLoader stall time: 3.16 ms/step |
| opt-003 | Use Fused Optimizer Kernels | kernel_fusion | 1.1x | medium | Backward time: 17.67ms (70.0%); Backward/Forward ratio: 2.3x |

### opt-001: Enable AMP for Mixed Precision Training

**Type**: mixed_precision | **Risk**: low | **Estimated speedup**: 1.2x | **Memory delta**: -10%

Enabling AMP can reduce the computation time and memory usage by using lower precision (fp16) during training, which is particularly beneficial for backward passes.

**Evidence**:
- Backward time: 17.67ms (70.0%)
- Backward/Forward ratio: 2.3x
- Peak Memory: 31.1 MB

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

### opt-002: Increase DataLoader Workers

**Type**: data_loading | **Risk**: low | **Estimated speedup**: 1.1x | **Memory delta**: +0%

Increasing the number of DataLoader workers can improve data throughput and reduce potential data loading stalls.

**Evidence**:
- DataLoader throughput: 1265.8 samples/sec
- DataLoader stall time: 3.16 ms/step

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

### opt-003: Use Fused Optimizer Kernels

**Type**: kernel_fusion | **Risk**: medium | **Estimated speedup**: 1.1x | **Memory delta**: +0%

Switching to a fused optimizer can improve performance by reducing kernel launch overhead and improving memory access patterns.

**Evidence**:
- Backward time: 17.67ms (70.0%)
- Backward/Forward ratio: 2.3x

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-003.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-003.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

## Validation Results

### Summary

| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |
|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|
| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |
| torch.compile(max-autotune + CUDA Graphs) | FAILED | - | - | - | - | - | - | - |
| torch.compile(max-autotune-no-cudagraphs) | 0.40x | -59.7% | +112.6% | +148.4% | N/A | N/A | YES | 0.880198 |
| Enable AMP for Mixed Precision Training | 0.69x | -30.9% | +0.0% | +44.7% | -0.331 | N/A | No | 0.000000 |
| Increase DataLoader Workers | 1.04x | +3.7% | +0.0% | -3.6% | 0.033 | N/A | No | 0.000000 |
| Use Fused Optimizer Kernels | 0.62x | -37.6% | +0.0% | +60.4% | -0.422 | N/A | No | 0.000000 |


**Control run**: 83 steps, 1.3s, 16.0 ms/step, peak memory 31.1 MB

### Loss Curves

| Step | Control | torch.compile(max-au | Enable AMP for Mixed | Increase DataLoader  | Use Fused Optimizer  |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.7074 | 0.7074 | 0.7074 | 0.7074 | 0.7074 |
| 8 | 0.7020 | 0.7019 | 0.7020 | 0.7020 | 0.7020 |
| 16 | 0.6964 | 0.6964 | 0.6964 | 0.6964 | 0.6964 |
| 24 | 0.6971 | 0.6972 | 0.6971 | 0.6971 | 0.6971 |
| 32 | 0.6942 | 0.6944 | 0.6942 | 0.6942 | 0.6942 |
| 40 | 0.6941 | 0.6941 | 0.6941 | 0.6941 | 0.6941 |
| 48 | 0.6933 | 0.6934 | 0.6933 | 0.6933 | 0.6933 |
| 56 | 0.6933 | 0.6934 | 0.6933 | 0.6933 | 0.6933 |
| 64 | 0.6931 | 0.6931 | 0.6931 | 0.6931 | 0.6931 |
| 72 | 0.6934 | 0.6934 | 0.6934 | 0.6934 | 0.6934 |
| 80 | 0.6934 | 0.6934 | 0.6934 | 0.6934 | 0.6934 |

### Gradient Similarity

| Step | torch.compile(max-au | Enable AMP for Mixed | Increase DataLoader  | Use Fused Optimizer  | Threshold |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 8 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 16 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 24 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 32 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 40 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 48 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 56 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 64 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 72 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |
| 80 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 |

### Correctness (Logit Signatures)

- **torch.compile(max-autotune-no-cudagraphs)**: checks=83, max_abs_diff=0.880198, mean_abs_diff=0.369561 (exceeded tolerance)
- **Enable AMP for Mixed Precision Training**: checks=83, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
- **Increase DataLoader Workers**: checks=83, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
- **Use Fused Optimizer Kernels**: checks=83, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
### Divergence Flags

- **torch.compile(max-autotune-no-cudagraphs)**: Diverged at step 1. Reason: Logit signature delta 0.880198 exceeded tolerance 0.001000

## Evidence Tracebacks

### Evidence Graph

- `Native baseline candidate required by benchmark ladder.` -> `B1` (supports)
- `Detected GPU count: 1` -> `B1` (supports)
- `Fallback mode: default` -> `B1` (supports)
- `Backward time: 17.67ms (70.0%)` -> `opt-001` (supports)
- `Backward/Forward ratio: 2.3x` -> `opt-001` (supports)
- `Peak Memory: 31.1 MB` -> `opt-001` (supports)
- `DataLoader throughput: 1265.8 samples/sec` -> `opt-002` (supports)
- `DataLoader stall time: 3.16 ms/step` -> `opt-002` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-002` (supports)
- `Backward time: 17.67ms (70.0%)` -> `opt-003` (supports)
- `Backward/Forward ratio: 2.3x` -> `opt-003` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-003` (supports)

### Increase DataLoader Workers

**Baseline comparison**
- vs control speedup: 1.04x
- vs control cost delta: -3.6%
- vs control throughput delta: +3.7%

**Profiler traceback**
- DataLoader throughput: 1265.8 samples/sec
- DataLoader stall time: 3.16 ms/step
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 83
- Logit max abs diff: 0.000000
- Objective score: 0.033

**Code traceback**
- No code patch required (runtime/config optimization).
### Enable AMP for Mixed Precision Training

**Baseline comparison**
- vs control speedup: 0.69x
- vs control cost delta: +44.7%
- vs control throughput delta: -30.9%

**Profiler traceback**
- Backward time: 17.67ms (70.0%)
- Backward/Forward ratio: 2.3x
- Peak Memory: 31.1 MB
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 83
- Logit max abs diff: 0.000000
- Objective score: -0.331

**Code traceback**
- No code patch required (runtime/config optimization).
### Use Fused Optimizer Kernels

**Baseline comparison**
- vs control speedup: 0.62x
- vs control cost delta: +60.4%
- vs control throughput delta: -37.6%

**Profiler traceback**
- Backward time: 17.67ms (70.0%)
- Backward/Forward ratio: 2.3x
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 83
- Logit max abs diff: 0.000000
- Objective score: -0.422

**Code traceback**
- No code patch required (runtime/config optimization).

## Native Baseline Ladder

| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |
|----------|----------|---------|--------------------|--------------|--------|-----------|
| B0 (Control (eager, no optimization)) | YES | YES | - | - | - | - |
| B1 (torch.compile(max-autotune-no-cudagraphs)) | YES | No | 0.40x | -59.7% | +148.4% | - |
| B2 (torch.compile(max-autotune + CUDA Graphs)) | No | No | - | - | - | - |

- `B2` skipped: Reserved for multi-GPU cluster runs

## Recommendations

1. **Consider**: Increase DataLoader Workers -- 1.0x speedup, +0% memory
2. **Skip**: torch.compile(max-autotune + CUDA Graphs) -- Failed to run
3. **Skip**: torch.compile(max-autotune-no-cudagraphs) -- Diverged: Logit signature delta 0.880198 exceeded tolerance 0.001000
4. **Skip**: Enable AMP for Mixed Precision Training -- Slower than baseline
5. **Skip**: Use Fused Optimizer Kernels -- Slower than baseline

## Appendix


### Raw Config Data

**opt-001: Enable AMP for Mixed Precision Training**

```json
{
  "id": "opt-001",
  "name": "Enable AMP for Mixed Precision Training",
  "description": "Enabling AMP can reduce the computation time and memory usage by using lower precision (fp16) during training, which is particularly beneficial for backward passes.",
  "optimization_type": "mixed_precision",
  "evidence": [
    "Backward time: 17.67ms (70.0%)",
    "Backward/Forward ratio: 2.3x",
    "Peak Memory: 31.1 MB"
  ],
  "code_changes": {},
  "config_overrides": {},
  "estimated_speedup": 1.2,
  "estimated_memory_delta": -0.1,
  "risk_level": "low",
  "dependencies": [],
  "is_native_baseline": false,
  "baseline_id": null,
  "eligible": true,
  "ineligible_reason": null,
  "heuristic_rationale": []
}
```

**opt-002: Increase DataLoader Workers**

```json
{
  "id": "opt-002",
  "name": "Increase DataLoader Workers",
  "description": "Increasing the number of DataLoader workers can improve data throughput and reduce potential data loading stalls.",
  "optimization_type": "data_loading",
  "evidence": [
    "DataLoader throughput: 1265.8 samples/sec",
    "DataLoader stall time: 3.16 ms/step"
  ],
  "code_changes": {},
  "config_overrides": {},
  "estimated_speedup": 1.05,
  "estimated_memory_delta": 0.0,
  "risk_level": "low",
  "dependencies": [],
  "is_native_baseline": false,
  "baseline_id": null,
  "eligible": true,
  "ineligible_reason": null,
  "heuristic_rationale": []
}
```

**opt-003: Use Fused Optimizer Kernels**

```json
{
  "id": "opt-003",
  "name": "Use Fused Optimizer Kernels",
  "description": "Switching to a fused optimizer can improve performance by reducing kernel launch overhead and improving memory access patterns.",
  "optimization_type": "kernel_fusion",
  "evidence": [
    "Backward time: 17.67ms (70.0%)",
    "Backward/Forward ratio: 2.3x"
  ],
  "code_changes": {},
  "config_overrides": {},
  "estimated_speedup": 1.15,
  "estimated_memory_delta": 0.0,
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