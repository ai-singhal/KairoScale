# GPUnity Optimization Report

> **Repo**: `/Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet` | **GPU**: a100-80gb | **Workload**: train | **Objective**: balanced | **Date**: 2026-02-28 19:09 | **Mode**: modal


## Executive Summary

- **Top proposed config**: "Enable Data Parallelism" -- estimated 1.5x speedup
- 0/0 configs validated successfully
- 0/0 configs executed without runtime failure
- No divergence detected in any configuration

> *Note: This is a dry-run report. Validation was skipped.*

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
- DataLoader stall: 12.80 ms/step
- Training loop detection: none

> **Warning**: Training loop could not be automatically detected. Consider using `--train-function <name>` for more accurate profiling.

### Operator Breakdown

*No operator data available.*


## Proposed Optimizations

| ID | Name | Type | Est. Speedup | Risk | Evidence |
|----|------|------|-------------|------|----------|
| opt-002 | Enable Data Parallelism | parallelism | 1.5x | low | GPU Utilization: 0.0% (indicating potential underutilization); Code supports DataParallel when multiple GPUs are detected. |
| opt-001 | Enable Mixed Precision (AMP) | mixed_precision | 1.2x | low | Backward time: 99.33ms (70.0%); Forward/Backward ratio: 2.3x |

### opt-002: Enable Data Parallelism

**Type**: parallelism | **Risk**: low | **Estimated speedup**: 1.5x | **Memory delta**: +0%

The current setup supports multiple GPUs via DataParallel. If not already configured, enabling this can distribute the workload across available GPUs, improving throughput.

**Evidence**:
- GPU Utilization: 0.0% (indicating potential underutilization)
- Code supports DataParallel when multiple GPUs are detected.

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-002.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

### opt-001: Enable Mixed Precision (AMP)

**Type**: mixed_precision | **Risk**: low | **Estimated speedup**: 1.2x | **Memory delta**: -20%

The backward pass is significantly longer than the forward pass, indicating potential benefits from mixed precision training. Using Automatic Mixed Precision (AMP) can reduce memory usage and speed up training by using FP16 where possible, while maintaining FP32 precision for critical operations.

**Evidence**:
- Backward time: 99.33ms (70.0%)
- Forward/Backward ratio: 2.3x

**Config JSON**: `/Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json`
**Apply command**: `gpunity apply /Users/aryan/Downloads/AutoProfile/logs/gpunity_configs/opt-001.json --repo /Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet`

## Validation Results

*Skipped (dry-run mode).*


## Recommendations

Validation was not run. Review the proposed optimizations above and re-run without `--dry-run` to validate them.

1. **Consider**: Enable Data Parallelism -- estimated 1.5x speedup (low risk)
2. **Consider**: Enable Mixed Precision (AMP) -- estimated 1.2x speedup (low risk)

## Appendix


### Raw Config Data

**opt-002: Enable Data Parallelism**

```json
{
  "id": "opt-002",
  "name": "Enable Data Parallelism",
  "description": "The current setup supports multiple GPUs via DataParallel. If not already configured, enabling this can distribute the workload across available GPUs, improving throughput.",
  "optimization_type": "parallelism",
  "evidence": [
    "GPU Utilization: 0.0% (indicating potential underutilization)",
    "Code supports DataParallel when multiple GPUs are detected."
  ],
  "code_changes": {},
  "config_overrides": {},
  "estimated_speedup": 1.5,
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

**opt-001: Enable Mixed Precision (AMP)**

```json
{
  "id": "opt-001",
  "name": "Enable Mixed Precision (AMP)",
  "description": "The backward pass is significantly longer than the forward pass, indicating potential benefits from mixed precision training. Using Automatic Mixed Precision (AMP) can reduce memory usage and speed up training by using FP16 where possible, while maintaining FP32 precision for critical operations.",
  "optimization_type": "mixed_precision",
  "evidence": [
    "Backward time: 99.33ms (70.0%)",
    "Forward/Backward ratio: 2.3x"
  ],
  "code_changes": {},
  "config_overrides": {},
  "estimated_speedup": 1.2,
  "estimated_memory_delta": -0.2,
  "risk_level": "low",
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