# GPUnity — ML Training Optimization Pipeline

## Overview

GPUnity is a CLI tool that takes an ML training repo, profiles it in a cloud sandbox, uses an LLM agent to generate optimization configurations grounded in profile evidence, validates the top configs in parallel sandboxes while tracking gradient divergence, and produces a final report with speed/cost/memory deltas.

```
gpunity run ./my-repo --entry train.py --steps 50 --provider claude
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                        gpunity run ./my-repo                        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │   PHASE 1: PROFILE        │
                 │   Modal sandbox            │
                 │   • torch.profiler         │
                 │   • Memory snapshots       │
                 │   • Autograd profiler      │
                 │   • DataLoader stats       │
                 └─────────────┬──────────────┘
                               │ ProfileResult
                               ▼
                 ┌──────────────────────────┐
                 │   PHASE 2: ANALYZE        │
                 │   LLM Agent (pluggable)   │
                 │   • Reads profiles + code  │
                 │   • Proposes k configs     │
                 │   • Selects top n diverse  │
                 └─────────────┬──────────────┘
                               │ OptimizationConfig[]
                               ▼
                 ┌──────────────────────────┐
                 │   PHASE 3: VALIDATE       │
                 │   n parallel sandboxes    │
                 │   + 1 control run          │
                 │   • m steps each           │
                 │   • Track grad cos sim     │
                 │   • Track loss, memory     │
                 │   • Detect divergence      │
                 └─────────────┬──────────────┘
                               │ ValidationResult[]
                               ▼
                 ┌──────────────────────────┐
                 │   PHASE 4: REPORT         │
                 │   Markdown generation      │
                 │   • Bottleneck analysis    │
                 │   • Config comparisons     │
                 │   • Speed/cost/mem deltas  │
                 │   • Divergence flags       │
                 └──────────────────────────┘
```

---

## Phase 1: Profile

### Goal

Instrument the user's training script with every profiler we have and capture a rich set of artifacts that the agent can reason over.

### Execution Environment

The user's repo is mounted into a **Modal sandbox** with GPU (user-configurable, default A100-80GB). Modal gives us:

- Deterministic, reproducible environments (Docker image + pip deps from user's `requirements.txt`)
- On-demand GPU access with per-second billing
- File I/O for extracting profile artifacts back to the host

### Instrumentation Strategy

We wrap the user's training entry point rather than modifying their code. The wrapper:

1. Imports the user's training script as a module
2. Monkey-patches `torch.utils.data.DataLoader` to add throughput/stall tracking
3. Wraps the training loop (detected via heuristics or user annotation) with profilers
4. Runs $w$ warmup steps (unprofiled), then $p$ profiled steps

### Profilers

**torch.profiler (operator + CUDA kernels)**

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=w, active=p, repeat=1),
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    on_trace_ready=trace_handler,
) as prof:
    for step in training_loop:
        ...
        prof.step()
```

Outputs: Chrome trace JSON, `key_averages()` table sorted by CUDA time. We extract the top operators, their % of total GPU time, call counts, and input shapes.

**Memory snapshots**

```python
torch.cuda.memory._record_memory_history(max_entries=100000)
# ... run steps ...
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._record_memory_history(enabled=None)
```

Outputs: Full allocation timeline with stack traces. We extract peak memory, the allocation that caused the peak, and a timeline of memory usage per step. This tells the agent whether memory is the bottleneck and where the big allocations are.

**Autograd profiler (backward pass)**

```python
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    loss.backward()
```

Outputs: Forward vs backward time split, operator-level breakdown of the backward pass. Critical for identifying whether the backward pass is disproportionately slow (suggesting gradient checkpointing opportunities).

**DataLoader throughput stats**

We wrap `DataLoader.__iter__` to measure:

- Time spent waiting for the next batch (stall time)
- Effective throughput (samples/sec)
- Whether the GPU is starved (stall time > compute time per step)

This is a custom wrapper, not a PyTorch built-in. We measure wall-clock time between `next(dataloader_iter)` calls vs. time spent in the forward/backward pass.

### Profile Artifact Format

All profiler outputs get aggregated into a `ProfileResult` dataclass:

```
ProfileResult:
  # torch.profiler
  top_operators: [{name, gpu_time_ms, cpu_time_ms, pct_total, call_count, flops}]
  gpu_utilization: float (% of step time spent in CUDA kernels)
  chrome_trace_path: Path

  # Memory
  peak_memory_mb: float
  memory_timeline: [{step, allocated_mb, reserved_mb}]
  peak_allocation_stack: str

  # Autograd
  forward_time_ms: float
  backward_time_ms: float
  backward_ops: [{name, time_ms, pct_backward}]

  # DataLoader
  dataloader_throughput: float (samples/sec)
  dataloader_stall_time_ms: float (per step avg)
  dataloader_bottleneck: bool
```

### Open Questions — Phase 1

- **Training loop detection**: How do we find the training loop if the user doesn't annotate it? Heuristics: look for `loss.backward()`, `optimizer.step()` patterns. Fallback: require `# gpunity:train_start` / `# gpunity:train_end` comments or a config flag `--train-function main_loop`.
- **Multi-GPU profiling**: Do we profile on a single GPU first, then extrapolate? Or profile on the target parallelism config? Starting with single-GPU profiling is simpler and captures per-device bottlenecks.
- **Profile duration**: How many steps is enough? Default 20 should be sufficient for steady-state behavior, but we should skip the first few steps due to CUDA lazy init and JIT warmup.

---

## Phase 2: Analyze & Generate

### Goal

An LLM agent reads the profile artifacts + the user's source code and proposes $k$ optimization configurations. Each config is grounded in specific profile evidence — no hallucinated suggestions. The evidence should be visible in the final report (I cannot emphasize this enough).

### Agent Architecture

The agent is a loop, not a single prompt. It has access to tools:

```
Tools:
  read_profile()     → returns ProfileResult.summary()
  read_file(path)    → reads a file from the user's repo
  list_files()       → lists the repo structure
  search_code(query) → grep-like search across the repo
  propose_config()   → structured output: OptimizationConfig
  estimate_impact()  → rough speedup/memory estimate given a config
```

**Agent loop** (max $I$ iterations):

```
1. Agent reads the profile summary
2. Agent identifies bottlenecks (e.g., "attention ops = 58% of GPU time")
3. Agent reads relevant source files to understand the current implementation
4. Agent proposes an optimization config with:
   - What to change (code patches or config overrides)
   - Why (grounded in profile data)
   - Estimated impact
   - Risk level
5. Repeat until k configs generated or agent decides it has enough
```

### Pluggable Providers

The agent logic is provider-agnostic. We define an `LLMProvider` interface:

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[Tool],
        temperature: float = 0.3,
    ) -> CompletionResult: ...
```

Built-in providers:

| Provider | Model | Env Var |
|----------|-------|---------|
| `claude` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| `custom` | user-specified | user-specified |

The CLI flag is `--provider claude` (default) or `--provider openai` or `--provider custom --model-url https://...`.

### Optimization Categories

The agent is prompted to consider these categories and ONLY suggest optimizations it can justify from the profile data:

| Category | Example Optimizations | Profile Signal |
|----------|----------------------|----------------|
| Attention | Flash Attention, memory-efficient attention | High % GPU time in `aten::scaled_dot_product_attention` or manual attention ops |
| Compilation | `torch.compile`, inductor backends | Many small kernels, low GPU utilization, high kernel launch overhead |
| Mixed Precision | AMP, bf16, fp8 | Already in fp32, memory-bound ops dominating |
| Data Loading | Pre-tokenization, more workers, prefetch | High `dataloader_stall_time`, `dataloader_bottleneck=True` |
| Parallelism | FSDP, tensor parallel | Single-GPU, memory near capacity, model too large |
| Memory | Gradient checkpointing, activation offload | Peak memory near GPU limit, large activation tensors |
| Kernel Fusion | Custom fused ops, `torch.compile` | Many small sequential ops that could be fused |
| Communication | Overlap compute/comm, gradient bucketing | Multi-GPU with comm overhead (future) |

### Diversity Selection

After generating $k$ configs, we select the top $n$ for validation using a diversity-aware ranking:

1. Score each config by `estimated_speedup * (1 / risk_level_weight)`
2. Greedily select configs, penalizing configs that share the same `optimization_type` as already-selected ones
3. Ensure at least one config from each distinct bottleneck category identified in the profile

The diversity threshold $\delta$ controls how aggressively we penalize similar configs. The point is to avoid validating 5 variations of "use Flash Attention" when we should also test compilation, mixed precision, etc.

### Output

A ranked list of `OptimizationConfig` objects, each containing:

```
OptimizationConfig:
  id: str                          # "opt-001"
  name: str                        # "Flash Attention + bf16"
  description: str                 # Human-readable explanation
  optimization_type: enum          # ATTENTION, COMPILATION, etc.
  evidence: list[str]              # ["sdpa accounts for 58% GPU time", ...]
  code_changes: dict[str, str]     # {file_path: unified_diff}
  config_overrides: dict           # {"precision": "bf16", "compile": true}
  estimated_speedup: float         # 2.1x
  estimated_memory_delta: float    # -0.35 (35% reduction)
  risk_level: str                  # "low" | "medium" | "high"
  dependencies: list[str]          # ["flash-attn>=2.5"]
```

### Open Questions — Phase 2

- **Code patching strategy**: Should the agent output unified diffs, or full file replacements? Diffs are more token-efficient but harder to apply reliably. Leaning toward full file replacements for changed files.
- **Config-only vs code changes**: Some optimizations are pure config (set `torch.compile(model)` or `precision=bf16`). Others require code changes (swap attention implementation). The agent should prefer config-only changes when possible since they're lower risk.
- **Agent calibration**: How do we prevent the agent from being overconfident in estimated speedups? Include few-shot examples of realistic estimates, and emphasize that validation will catch bad estimates.

---

## Phase 3: Validate

### Goal

Run each selected config for $m$ steps in parallel Modal sandboxes, alongside a **control run** (unmodified code), and compare performance + numerical stability.

### Execution Plan

```
Parallel sandbox launches:
  sandbox_0: CONTROL (original code, no modifications)
  sandbox_1: config "opt-001" (Flash Attention + bf16)
  sandbox_2: config "opt-002" (torch.compile + grad checkpointing)
  sandbox_3: config "opt-003" (optimized dataloader + bf16)
  ...
  sandbox_n: config "opt-00n"
```

Each sandbox:

1. Clones the repo
2. Applies code changes from the config (patches or file replacements)
3. Installs any additional dependencies
4. Seeds everything deterministically (`torch.manual_seed`, `torch.cuda.manual_seed_all`, etc.)
5. Runs $m$ training steps with instrumentation

### Metrics Collected Per Sandbox

**Performance:**

- Wall-clock time for $m$ steps
- Per-step time (mean, p50, p95, p99)
- Peak GPU memory
- Throughput (samples/sec)

**Cost:**

- GPU-seconds consumed
- Estimated $ cost (from Modal's pricing: e.g., A100-80GB at ~$X/hr)
- Cost delta vs control

**Numerical Stability (the critical part):**

Every $g$ steps (default $g = 5$), we checkpoint:

- The **gradient tensor** of a reference parameter (e.g., the first attention layer's QKV weight)
- The **loss value**

We compute:

$$\text{cos\_sim}(t) = \frac{\nabla_{\theta}^{\text{opt}}(t) \cdot \nabla_{\theta}^{\text{ctrl}}(t)}{\|\nabla_{\theta}^{\text{opt}}(t)\| \cdot \|\nabla_{\theta}^{\text{ctrl}}(t)\|}$$

where $\nabla_{\theta}^{\text{opt}}(t)$ is the gradient at step $t$ for the optimized config and $\nabla_{\theta}^{\text{ctrl}}(t)$ is the control's gradient.

**Divergence detection rules:**

1. If $\text{cos\_sim}(t) < \tau$ (default $\tau = 0.8$) for 3 consecutive checks → flag as **diverged**
2. If loss ratio $\frac{L_{\text{opt}}(t)}{L_{\text{ctrl}}(t)} > 2.0$ → flag as **diverged**
3. If loss is `NaN` or `Inf` → flag as **crashed**

### Gradient Synchronization Problem

For the cosine similarity check to be meaningful, the control and optimized runs must:

- Start from the **exact same weights** (same checkpoint)
- Use the **exact same data** in the same order (same seed, same dataloader shuffle)
- Differ ONLY in the optimization applied

This means we need to:

1. Save a checkpoint before validation begins (after Phase 1 profiling)
2. Ship this checkpoint to all sandboxes
3. Ensure the dataloader is deterministic (seed all RNGs, set `num_workers=0` or use `worker_init_fn` with fixed seeds)

### Implementation Detail: Gradient Hooks

We can't easily extract gradients from an arbitrary training script without modification. The approach:

```python
# Injected into the user's training script
reference_param = find_reference_param(model)  # heuristic: largest attention weight

gradients = []
def grad_hook(grad):
    if step % gradient_check_interval == 0:
        gradients.append(grad.detach().clone().cpu())
    return grad

reference_param.register_hook(grad_hook)
```

The `find_reference_param` heuristic looks for parameters with "attn", "attention", "qkv", or "self_attn" in the name. Fallback: the largest parameter by numel.

### Output

```
ControlRun:
  steps_completed: int
  wall_clock_seconds: float
  avg_step_time_ms: float
  peak_memory_mb: float
  throughput_samples_sec: float
  loss_values: list[float]
  gradient_norms: list[float]
  cost_estimate_usd: float

ValidationResult (per config):
  config_id: str
  success: bool
  error: str (if failed)

  # Performance deltas
  speedup_vs_control: float        # 1.8x
  memory_delta_vs_control: float   # -0.25 (25% less)
  cost_delta_vs_control: float     # -0.40 (40% cheaper)

  # Stability
  gradient_cosine_similarities: list[float]  # per check
  loss_values: list[float]
  diverged: bool
  divergence_step: int | None
  divergence_reason: str
```

### Open Questions — Phase 3

- **How many steps?** Default $m = 50$. Enough to see steady-state behavior and catch divergence, but not so many that validation is expensive. This is configurable.
- **Which gradient to track?** Tracking ALL gradients is too expensive. We pick one reference parameter. But what if the divergence happens in a different layer? Could do a cheap check: compare final loss curves. If loss diverges, that catches most issues. The gradient cosine sim is a finer-grained early warning.
- **What if the user's code doesn't have a clean training loop?** We need to either detect it or require annotation. Same issue as Phase 1.
- **Cost control**: $n$ parallel A100 sandboxes for $m$ steps each. At 50 steps, this should be minutes per sandbox. But we should add a cost ceiling / timeout. Default: abort if any single sandbox exceeds $5 or 10 minutes.

---

## Phase 4: Report

### Goal

Generate a polished Markdown file that a team lead or ML engineer can read to decide which optimizations to adopt.

### Report Structure

```markdown
# GPUnity Optimization Report
> Repo: ./my-repo | GPU: A100-80GB | Date: 2025-02-28

## Executive Summary
- Best config: "Flash Attention + bf16" → 2.1x speedup, 35% less memory
- X of Y configs validated successfully
- No divergence detected in top recommendation
- Estimated annual savings: $XX,XXX at current training scale

## Profile Analysis
### Bottleneck Breakdown
[Pie chart or table showing GPU time by operator category]

### Key Findings
- Attention ops consume 58% of GPU time
- Peak memory at 71.2GB / 80GB (89% utilization)
- DataLoader is NOT a bottleneck (stall < 2% of step time)
- Forward/backward ratio: 1:2.8 (backward-heavy, checkpointing may help)

## Proposed Optimizations
[Table: config name | type | evidence | est. speedup | risk]

## Validation Results

### Summary Table
| Config | Speedup | Memory Δ | Cost Δ | Diverged? | Avg Cos Sim |
|--------|---------|----------|--------|-----------|-------------|
| Flash Attn + bf16 | 2.1x | -35% | -52% | No | 0.97 |
| torch.compile | 1.4x | -5% | -29% | No | 0.99 |
| Grad checkpointing | 0.9x | -45% | +11% | No | 0.98 |

### Per-Config Details
[For each config: loss curve comparison, gradient cosine sim over steps, memory timeline]

### Divergence Flags
[Any configs that diverged, with the step and reason]

## Recommendations
1. **Adopt**: Flash Attention + bf16 — 2.1x speedup with no divergence
2. **Consider**: torch.compile — 1.4x speedup, minimal risk
3. **Skip**: Grad checkpointing — slower runtime despite memory savings

## Appendix
- Full profile operator table
- Raw config YAML for each optimization
- Cost breakdown
```

### Charts

The Markdown report includes inline charts rendered as either:

- ASCII tables (always works)
- Embedded base64 PNG charts generated via `matplotlib` (richer, but requires rendering)

Default: ASCII tables with an option `--charts png` to generate matplotlib charts.

### Open Questions — Phase 4

- **Interactive report?** Could generate an HTML report with interactive charts (Plotly). Nice to have, not MVP.
- **Config export**: Should the report include copy-pasteable code changes? Yes — each config section should have the exact diff or config YAML to apply.

---

## CLI Design

```
gpunity run <repo_path>
  --entry <file>              Training script entry point (default: train.py)
  --train-function <name>     Function containing the training loop (optional)
  --provider <name>           LLM provider: claude|openai|custom (default: claude)
  --model <name>              Model override for the provider
  --gpu <type>                GPU type: a100-80gb|a100-40gb|h100|a10g (default: a100-80gb)
  --profile-steps <n>         Steps to profile (default: 20)
  --validation-steps <n>      Steps per validation run (default: 50)
  --max-configs <k>           Total configs to generate (default: 10)
  --top-k <n>                 Configs to validate (default: 5)
  --divergence-threshold <τ>  Cosine sim threshold (default: 0.8)
  --max-cost <$>              Cost ceiling per sandbox (default: 5.00)
  --output <path>             Report output path (default: ./gpunity_report.md)
  --charts <mode>             Chart mode: ascii|png (default: ascii)
  --verbose                   Show detailed logs
  --dry-run                   Run Phase 1-2 only, skip validation
```

```
gpunity profile <repo_path>    # Phase 1 only
gpunity analyze <profile_dir>  # Phase 2 only (from saved profiles)
gpunity validate <config_dir>  # Phase 3 only (from saved configs)
```

---

## Project Structure

```
gpunity/
├── cli.py                    # Click CLI entry point
├── types.py                  # Shared dataclasses (ProfileResult, OptimizationConfig, etc.)
├── config.py                 # YAML config + CLI arg merging
│
├── profiler/
│   ├── __init__.py
│   ├── wrapper.py            # Training script wrapper + monkey-patching
│   ├── torch_profiler.py     # torch.profiler setup + extraction
│   ├── memory.py             # CUDA memory snapshot handling
│   ├── autograd.py           # Autograd profiler for fwd/bwd split
│   ├── dataloader.py         # DataLoader throughput instrumentation
│   └── aggregate.py          # Combine all profiler outputs → ProfileResult
│
├── sandbox/
│   ├── __init__.py
│   ├── modal_runner.py       # Modal sandbox creation + execution
│   ├── image_builder.py      # Docker image construction from user's deps
│   └── artifact_io.py        # Upload/download artifacts to/from sandbox
│
├── agent/
│   ├── __init__.py
│   ├── loop.py               # Main agent loop (iterate, propose, select)
│   ├── tools.py              # Agent tool definitions (read_profile, read_file, etc.)
│   ├── prompts.py            # System prompts + few-shot examples
│   ├── diversity.py          # Diversity-aware config selection
│   └── providers/
│       ├── __init__.py
│       ├── base.py           # LLMProvider protocol
│       ├── claude.py         # Anthropic API provider
│       ├── openai.py         # OpenAI API provider
│       └── custom.py         # Generic OpenAI-compatible provider
│
├── validator/
│   ├── __init__.py
│   ├── runner.py             # Orchestrate parallel validation runs
│   ├── patcher.py            # Apply code changes from OptimizationConfig
│   ├── gradient_tracker.py   # Gradient hooks + cosine similarity
│   ├── divergence.py         # Divergence detection logic
│   └── metrics.py            # Collect wall-clock, memory, throughput
│
├── reporter/
│   ├── __init__.py
│   ├── markdown.py           # Markdown report generation
│   ├── charts.py             # ASCII + matplotlib chart generation
│   └── templates/            # Jinja2 templates for report sections
│
└── utils/
    ├── logging.py
    ├── cost.py               # GPU cost estimation
    └── repo.py               # Repo scanning, dep detection
```

---

## Implementation Priority

**Phase 0 — Scaffolding** (do first):

- `types.py` — all shared dataclasses
- `config.py` — YAML + CLI config
- `cli.py` — Click CLI skeleton
- Modal sandbox hello-world (spin up, run a script, pull artifacts back)

**Phase 1 — Profile** (core value):

- `profiler/wrapper.py` — the wrapper that instruments user code
- `profiler/torch_profiler.py` + `profiler/memory.py` — these two profilers give 80% of the value
- `sandbox/modal_runner.py` — run the wrapped script in Modal
- Test on a known training script (e.g., nanoGPT)

**Phase 2 — Agent** (differentiator):

- `agent/providers/base.py` + `agent/providers/claude.py` — get one provider working
- `agent/prompts.py` — the system prompt is the most important part
- `agent/loop.py` — agent loop with tool use
- `agent/diversity.py` — config selection

**Phase 3 — Validate** (trust layer):

- `validator/patcher.py` — apply code changes
- `validator/gradient_tracker.py` — gradient hooks
- `validator/runner.py` — parallel sandbox orchestration
- `validator/divergence.py` — detection logic

**Phase 4 — Report** (output):

- `reporter/markdown.py` — Jinja2 templates + data formatting
- `reporter/charts.py` — ASCII tables first, matplotlib later

---

## Key Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training loop detection fails | Can't instrument code | Require user annotation as fallback (`--train-function`) |
| Agent generates invalid code patches | Validation crashes | Sandbox isolation + timeout; report error, don't crash pipeline |
| Gradient cosine sim is noisy | False divergence flags | Use rolling average over 3 checks; tune threshold |
| Modal cold start is slow | Pipeline takes too long | Pre-warm images; cache base images across runs |
| User's code has non-standard deps | Sandbox build fails | Parse `requirements.txt`, `setup.py`, `pyproject.toml`; fallback to user-provided Dockerfile |
| Cost overruns from parallel sandboxes | Expensive validation | Hard cost ceiling per sandbox; kill after timeout |
| Profiling overhead distorts results | Misleading profile data | Use PyTorch's built-in low-overhead profiling; separate warmup phase |

---

## Open Decisions Needed

1. **Training loop detection** — heuristic vs. required annotation vs. both?
2. **Code patching format*