# GPUnity Decision Log

## D-001: Training-Loop Annotation Fallback Semantics
- **Date**: 2026-02-28
- **Context**: When the profiler cannot determine training-loop boundaries (missing CUDA events, no profiler hooks, ambiguous op boundaries), the system must handle this gracefully. Three options considered: (A) conservative null, (B) heuristic inference, (C) user prompt.
- **Decision**: Option B -- Heuristic inference with fallback to A.
  - Primary: detect training loop via AST pattern matching for `loss.backward()` + `optimizer.step()` co-occurrence within a loop body.
  - If heuristic succeeds: use detected boundaries, flag `loop_detection: "heuristic"`, `confidence: "medium"` in ProfileResult.
  - If heuristic fails: fall back to Option A -- set `training_loop_annotation: null`, skip loop-level profiling, include a warning in the report, and suggest the user add `--train-function <name>` on next run.
  - Never fabricate loop boundaries or silently assume.
- **Rationale**: The nanoGPT fixture has a clear `loss.backward()` / `optimizer.step()` pattern, so heuristic inference keeps the fixture path green. For arbitrary user code, the fallback to null + warning is honest and actionable. Option C (user prompt) would block the pipeline and is inappropriate for a CLI tool that may run in CI.
- **Impact**: Affects `profiler/wrapper.py` (loop detection logic), `types.py` (add `loop_detection_method` and `loop_detection_confidence` fields to ProfileResult), `reporter/markdown.py` (add warning when annotation is null).

## D-002: Code Patching Format
- **Date**: 2026-02-28
- **Context**: Agent-proposed optimizations need to be applied to user code. Options: unified diffs vs full file replacements.
- **Decision**: Use `config_overrides` dict for config-only changes (preferred). For code changes, store full file content in `code_changes: dict[str, str]` (path -> new content). The validator/patcher writes these files directly.
- **Rationale**: Unified diffs are fragile with LLM-generated content. Full file replacements are reliable and the files being changed are typically small (single training scripts). Config-only changes are always preferred as lower risk.
- **Impact**: `agent/tools.py` (propose_config output format), `validator/patcher.py` (apply logic), `types.py` (OptimizationConfig schema).

## D-003: MVP Scope -- Modal Stubs for Local Testing
- **Date**: 2026-02-28
- **Context**: Modal API requires authentication and GPU access. The MVP must be testable without Modal credentials.
- **Decision**: All Modal-interacting code in `sandbox/` will have a `LocalRunner` alternative that runs profiling/validation in a local subprocess. The `--local` CLI flag selects it. The nanoGPT fixture tests use local mode exclusively. Modal integration is real code but tested separately.
- **Rationale**: Keeps the fixture path green without requiring cloud resources. Modal code is still production-quality, just not exercised in CI/fixture tests.
- **Impact**: `sandbox/modal_runner.py` (real Modal), `sandbox/local_runner.py` (subprocess fallback), `config.py` (add `--local` flag), `cli.py` (runner selection).
