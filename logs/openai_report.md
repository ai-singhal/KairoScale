# GPUnity Optimization Report

> **Repo**: `/Users/aryan/Downloads/AutoProfile/sample_codebases/nano_gpt` | **GPU**: h100 | **Workload**: train | **Objective**: balanced | **Date**: 2026-03-01 03:16 | **Mode**: modal


## Executive Summary

- **Best overall config ID**: `opt-001`
- **Best native baseline**: `B1`
- **Delta vs best native**: 1.03x speedup ratio, -2.6% cost delta, +2.6% throughput gain
- **Best config**: "Swap optimizer + Triton fused kernels" -- 1.0x speedup, +0% memory
- 2/4 configs validated successfully
- 3/4 configs executed without runtime failure
- 1 config(s) showed divergence
- Control run cost: $0.0630

## Hardware Context

- GPU: NVIDIA H100
- Detection source: declared-gpu (medium confidence)
- Compute capability: 9.0
- VRAM: 81920 MB
- Feature flags: compile=True, cuda_graphs=True, bf16=True, tf32=True
- Note: Runtime CUDA inspection unavailable; used gpu_type priors.

## Bottleneck Diagnosis

**Primary bottleneck**: Compute Bound (GPU Saturated)

**Evidence**:
- Insufficient profiler data for confident classification
- GPU utilization = 0.0%
- Peak memory = 0 MB
- DataLoader stall = 0.0 ms/step


## Profile Analysis

### Key Findings

- GPU Utilization: 0.0%
- Peak Memory: 0.0 MB
- Forward/Backward split: 30% / 70%
- Backward/Forward ratio: 2.3x
- DataLoader bottleneck: No
- Training loop detection: none

> **Warning**: Training loop could not be automatically detected. Consider using `--train-function <name>` for more accurate profiling.

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-002 | Stacked compile + AMP optimization combo | compilation | 1.2x | medium | Stacking compile + AMP for compound speedup on general workloads |
| opt-001 | Swap optimizer + Triton fused kernels | kernel_fusion | 1.1x | medium | Backward/forward ratio is 2.33x; High backward dominance can benefit from optimizer/kernel improvements |

### opt-002: Stacked compile + AMP optimization combo

**Type**: compilation | **Risk**: medium | **Estimated speedup**: 1.2x | **Memory delta**: -15%

Combines torch.compile (max-autotune) with bf16 mixed precision. These are broadly compatible and typically yield 15-30% combined speedup.

**Evidence**:
- Stacking compile + AMP for compound speedup on general workloads

**Config overrides**: `{'compile': True, 'compile_mode': 'max-autotune', 'amp': True, 'precision': 'bf16'}`

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/nano_gpt`

### opt-001: Swap optimizer + Triton fused kernels

**Type**: kernel_fusion | **Risk**: medium | **Estimated speedup**: 1.1x | **Memory delta**: -5%

Evaluate SGD/RMSProp/AdamW/Adafactor/MUON variants plus Triton-fused optimizer kernels where available.

**Evidence**:
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements

**Config overrides**: `{'optimizer_strategy': 'fused_triton_search', 'optimizer_candidates': ['sgd', 'rmsprop', 'adamw', 'adafactor', 'muon']}`

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/nano_gpt`

## Validation Results

### Summary

| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |
|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|
| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |
| torch.compile(max-autotune + CUDA Graphs) | FAILED | - | - | - | - | - | - | - |
| torch.compile(max-autotune-no-cudagraphs) | 0.99x | -0.8% | +73.3% | +0.8% | -0.010 | 1.00x | No | 0.000000 |
| Stacked compile + AMP optimization combo | 0.93x | -6.8% | +73.3% | +7.3% | N/A | N/A | YES | 20.787048 |
| Swap optimizer + Triton fused kernels | 1.02x | +1.8% | +0.0% | -1.7% | 0.009 | 1.03x | No | 0.000000 |


Best native baseline for this run: `B1`

**Control run**: 151 steps, 45.6s, 303.8 ms/step, peak memory 162.1 MB

### Loss Curves

| Step | Control | torch.compile(max-au | Stacked compile + AM | Swap optimizer + Tri |
| --- | --- | --- | --- | --- |
| 0 | 0.1046 | 0.1046 | 0.1046 | 0.1046 |
| 604 | 0.1032 | 0.1032 | 0.1037 | 0.1032 |
| 1208 | 0.0992 | 0.0992 | 0.1010 | 0.0992 |
| 1812 | 0.0946 | 0.0946 | 0.0971 | 0.0946 |
| 2416 | 0.0925 | 0.0925 | 0.0944 | 0.0925 |
| 3020 | 0.0901 | 0.0901 | 0.0920 | 0.0901 |
| 3624 | 0.0874 | 0.0874 | 0.0903 | 0.0874 |
| 4228 | 0.0861 | 0.0861 | 0.0895 | 0.0861 |
| 4832 | 0.0835 | 0.0835 | 0.0872 | 0.0835 |
| 5436 | 0.0821 | 0.0821 | 0.0854 | 0.0821 |

### Gradient Similarity

| Step | torch.compile(max-au | Stacked compile + AM | Swap optimizer + Tri | Threshold |
| --- | --- | --- | --- | --- |
| 0 | 1.000 | 1.000 | 1.000 | 0.80 |
| 604 | 1.000 | 0.993 | 1.000 | 0.80 |
| 1208 | 1.000 | 0.973 | 1.000 | 0.80 |
| 1812 | 1.000 | 0.963 | 1.000 | 0.80 |
| 2416 | 1.000 | 0.971 | 1.000 | 0.80 |
| 3020 | 1.000 | 0.970 | 1.000 | 0.80 |
| 3624 | 1.000 | 0.953 | 1.000 | 0.80 |
| 4228 | 1.000 | 0.944 | 1.000 | 0.80 |
| 4832 | 1.000 | 0.937 | 1.000 | 0.80 |
| 5436 | 1.000 | 0.943 | 1.000 | 0.80 |

### Correctness (Logit Signatures)

- **torch.compile(max-autotune-no-cudagraphs)**: checks=6040, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
- **Stacked compile + AMP optimization combo**: checks=6040, max_abs_diff=20.787048, mean_abs_diff=7.523605 (exceeded tolerance)
- **Swap optimizer + Triton fused kernels**: checks=6040, max_abs_diff=0.000000, mean_abs_diff=0.000000 (within tolerance)
### Divergence Flags

- **Stacked compile + AMP optimization combo**: Diverged at step 40. Reason: Logit signature delta 20.787048 exceeded tolerance 0.001000

## Evidence Tracebacks

### Evidence Graph

- `B1` -> `opt-001` (compared_against)
- `Native baseline candidate required by benchmark ladder.` -> `B1` (supports)
- `Detected GPU count: 1` -> `B1` (supports)
- `Fallback mode: default` -> `B1` (supports)
- `Stacking compile + AMP for compound speedup on general workloads` -> `opt-002` (supports)
- `Backward/forward ratio is 2.33x` -> `opt-001` (supports)
- `High backward dominance can benefit from optimizer/kernel improvements` -> `opt-001` (supports)
- `Ablation estimate: score combines speedup, throughput, cost reduction, and risk.` -> `opt-001` (supports)

### Swap optimizer + Triton fused kernels

**Baseline comparison**
- vs control speedup: 1.02x
- vs control cost delta: -1.7%
- vs control throughput delta: +1.8%
- vs best native (`B1`): 1.03x

**Profiler traceback**
- Backward/forward ratio is 2.33x
- High backward dominance can benefit from optimizer/kernel improvements
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 6040
- Logit max abs diff: 0.000000
- Objective score: 0.009

**Code traceback**
- No code patch required (runtime/config optimization).
### torch.compile(max-autotune-no-cudagraphs)

**Baseline comparison**
- vs control speedup: 0.99x
- vs control cost delta: +0.8%
- vs control throughput delta: -0.8%
- vs best native (`B1`): 1.00x

**Profiler traceback**
- Native baseline candidate required by benchmark ladder.
- Detected GPU count: 1
- Fallback mode: default
- Ablation estimate: score combines speedup, throughput, cost reduction, and risk.

**Validation block**
- Diverged: No
- Logit checks: 6040
- Logit max abs diff: 0.000000
- Objective score: -0.010

**Code traceback**
- No code patch required (runtime/config optimization).

## Native Baseline Ladder

| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |
|----------|----------|---------|--------------------|--------------|--------|-----------|
| B0 (Control (eager, no optimization)) | YES | YES | - | - | - | - |
| B1 (torch.compile(max-autotune-no-cudagraphs)) | YES | YES | 0.99x | -0.8% | +0.8% | -0.010 |
| B2 (torch.compile(max-autotune + CUDA Graphs)) | No | No | - | - | - | - |

- `B2` skipped: Reserved for multi-GPU cluster runs

## Recommendations

1. **Consider**: Swap optimizer + Triton fused kernels -- 1.0x speedup, +0% memory
2. **Skip**: torch.compile(max-autotune + CUDA Graphs) -- Failed to run
3. **Skip**: torch.compile(max-autotune-no-cudagraphs) -- Slower than baseline
4. **Skip**: Stacked compile + AMP optimization combo -- Diverged: Logit signature delta 20.787048 exceeded tolerance 0.001000

## Appendix


### Raw Config Data

**opt-002: Stacked compile + AMP optimization combo**

```json
{
  "id": "opt-002",
  "name": "Stacked compile + AMP optimization combo",
  "description": "Combines torch.compile (max-autotune) with bf16 mixed precision. These are broadly compatible and typically yield 15-30% combined speedup.",
  "optimization_type": "compilation",
  "evidence": [
    "Stacking compile + AMP for compound speedup on general workloads"
  ],
  "code_changes": {},
  "config_overrides": {
    "compile": true,
    "compile_mode": "max-autotune",
    "amp": true,
    "precision": "bf16"
  },
  "estimated_speedup": 1.25,
  "estimated_memory_delta": -0.15,
  "risk_level": "medium",
  "dependencies": [],
  "is_native_baseline": false,
  "baseline_id": null,
  "eligible": true,
  "ineligible_reason": null,
  "heuristic_rationale": [
    "Compound gains from stacking compile + AMP"
  ]
}
```

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