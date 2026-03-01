# GPUnity Optimization Report

> **Repo**: `/Users/aryan/Downloads/AutoProfile/sample_codebases/pytorch-3dunet` | **GPU**: a100-80gb | **Workload**: train | **Objective**: balanced | **Date**: 2026-02-28 19:08 | **Mode**: modal


## Executive Summary

- **Top proposed config**: "Swap optimizer + Triton fused kernels" -- estimated 1.1x speedup
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
- DataLoader stall: 15.74 ms/step
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

*Skipped (dry-run mode).*


## Recommendations

Validation was not run. Review the proposed optimizations above and re-run without `--dry-run` to validate them.

1. **Consider**: Swap optimizer + Triton fused kernels -- estimated 1.1x speedup (medium risk)
2. **Consider**: Enable selective gradient checkpointing -- estimated 0.9x speedup (medium risk)

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