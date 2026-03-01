# GPUnity Optimization Report

> **Repo**: `/Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/tests/fixtures/nano_gpt` | **GPU**: a100-80gb | **Workload**: infer | **Objective**: latency | **Date**: 2026-02-28 23:20 | **Mode**: local


## Executive Summary

- **Top proposed config**: "Stacked compile + AMP optimization combo" -- estimated 1.2x speedup
- 0/3 configs validated successfully
- 2/3 configs executed without runtime failure
- 2 config(s) showed divergence
- Control run cost: $0.0001

## Hardware Context

- GPU: NVIDIA A100 80GB
- Detection source: declared-gpu (medium confidence)
- Compute capability: 8.0
- VRAM: 81920 MB
- Feature flags: compile=True, cuda_graphs=True, bf16=True, tf32=True
- Note: Runtime CUDA inspection unavailable; used gpu_type priors.

## Bottleneck Diagnosis

**Primary bottleneck**: Compute Bound (GPU Saturated)

**Evidence**:
- Insufficient profiler data for confident classification
- GPU utilization = 0.0%
- Peak memory = 0 MB
- DataLoader stall = 0.1 ms/step


## Profile Analysis

### Key Findings

- GPU Utilization: 0.0%
- Peak Memory: 0.0 MB
- Forward/Backward split: 30% / 70%
- Backward/Forward ratio: 2.3x
- DataLoader bottleneck: No
- DataLoader stall: 0.07 ms/step
- Training loop detection: heuristic

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-002 | Stacked compile + AMP optimization combo | compilation | 1.2x | medium | Stacking compile + AMP for compound speedup on general workloads |

### opt-002: Stacked compile + AMP optimization combo

**Type**: compilation | **Risk**: medium | **Estimated speedup**: 1.2x | **Memory delta**: -15%

Combines torch.compile (max-autotune) with bf16 mixed precision. These are broadly compatible and typically yield 15-30% combined speedup.

**Evidence**:
- Stacking compile + AMP for compound speedup on general workloads

**Config overrides**: `{'compile': True, 'compile_mode': 'max-autotune', 'amp': True, 'precision': 'bf16'}`

**Config JSON**: `/Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/logs/gpunity_configs/opt-002.json`
**Apply command**: `gpunity apply /Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/logs/gpunity_configs/opt-002.json --repo /Users/manningwu/Desktop/personalRepos/Vibe_Coding/HackIllionis/AutoProfile/tests/fixtures/nano_gpt`

## Validation Results

### Summary

| Config | Speedup | Throughput Δ | Memory Delta | Cost Delta | Obj Score | vs Best Native | Diverged? | Logits Δmax |
|--------|---------|--------------|-------------|------------|-----------|----------------|-----------|------------|
| **Control** | 1.00x | baseline | baseline | baseline | - | - | - | - |
| torch.compile(max-autotune + CUDA Graphs) | FAILED | - | - | - | - | - | - | - |
| torch.compile(max-autotune-no-cudagraphs) | 0.01x | -99.3% | +0.0% | +14214.9% | N/A | N/A | YES | 0.483692 |
| Stacked compile + AMP optimization combo | 0.01x | -99.4% | +0.0% | +15600.6% | N/A | N/A | YES | 0.483157 |


**Control run**: 52 steps, 0.1s, 1.6 ms/step, peak memory 0.0 MB

### Loss Curves

| Step | Control | torch.compile(max-au | Stacked compile + AM |
| --- | --- | --- | --- |
| 0 | 5.6418 | 5.6462 | 5.6562 |
| 2 | 5.6709 | 5.6612 | 5.6250 |
| 4 | 5.6931 | 5.6920 | 5.7500 |
| 6 | 5.7100 | 5.7054 | 5.7188 |
| 8 | 5.7148 | 5.7180 | 5.6875 |
| 10 | 5.6712 | 5.6715 | 5.6562 |
| 12 | 5.6881 | 5.6918 | 5.6562 |
| 14 | 5.6800 | 5.6932 | 5.6875 |
| 16 | 5.6093 | 5.6101 | 5.5938 |
| 18 | 5.6045 | 5.6160 | 5.6562 |
| 20 | 5.6528 | 5.6650 | 5.6562 |
| 22 | 5.6308 | 5.6255 | 5.6562 |
| 24 | 5.6036 | 5.6169 | 5.6562 |

### Gradient Similarity

| Step | torch.compile(max-au | Stacked compile + AM | Threshold |
| --- | --- | --- | --- |
| 0 | 0.999 | 0.996 | 0.80 |
| 2 | 0.998 | 0.988 | 0.80 |
| 4 | 1.000 | 0.986 | 0.80 |
| 6 | 0.999 | 0.998 | 0.80 |
| 8 | 0.999 | 0.993 | 0.80 |
| 10 | 1.000 | 0.996 | 0.80 |
| 12 | 0.999 | 0.992 | 0.80 |
| 14 | 0.997 | 0.998 | 0.80 |
| 16 | 1.000 | 0.996 | 0.80 |
| 18 | 0.997 | 0.987 | 0.80 |
| 20 | 0.997 | 0.999 | 0.80 |
| 22 | 0.999 | 0.994 | 0.80 |
| 24 | 0.997 | 0.986 | 0.80 |

### Correctness (Logit Signatures)

- **torch.compile(max-autotune-no-cudagraphs)**: checks=26, max_abs_diff=0.483692, mean_abs_diff=0.288886 (exceeded tolerance)
- **Stacked compile + AMP optimization combo**: checks=26, max_abs_diff=0.483157, mean_abs_diff=0.288023 (exceeded tolerance)
### Divergence Flags

- **torch.compile(max-autotune-no-cudagraphs)**: Diverged at step 0. Reason: Logit signature delta 0.483692 exceeded tolerance 0.001000
- **Stacked compile + AMP optimization combo**: Diverged at step 0. Reason: Logit signature delta 0.483157 exceeded tolerance 0.001000

## Evidence Tracebacks

### Evidence Graph

- `Native baseline candidate required by benchmark ladder.` -> `B1` (supports)
- `Detected GPU count: 1` -> `B1` (supports)
- `Fallback mode: default` -> `B1` (supports)
- `Stacking compile + AMP for compound speedup on general workloads` -> `opt-002` (supports)

*No successful validated candidates for evidence tracebacks.*

## Native Baseline Ladder

| Baseline | Eligible | Success | Speedup vs Control | Throughput Δ | Cost Δ | Obj Score |
|----------|----------|---------|--------------------|--------------|--------|-----------|
| B0 (Control (eager, no optimization)) | YES | YES | - | - | - | - |
| B1 (torch.compile(max-autotune-no-cudagraphs)) | YES | No | 0.01x | -99.3% | +14214.9% | - |
| B2 (torch.compile(max-autotune + CUDA Graphs)) | No | No | - | - | - | - |

- `B2` skipped: Reserved for multi-GPU cluster runs

## Recommendations

1. **Skip**: torch.compile(max-autotune + CUDA Graphs) -- Failed to run
2. **Skip**: torch.compile(max-autotune-no-cudagraphs) -- Diverged: Logit signature delta 0.483692 exceeded tolerance 0.001000
3. **Skip**: Stacked compile + AMP optimization combo -- Diverged: Logit signature delta 0.483157 exceeded tolerance 0.001000

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


---
*Generated by GPUnity v0.1.0*