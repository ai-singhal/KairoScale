# GPUnity Interface Contracts (MVP)

This is the single source of truth for all module boundaries, function signatures, data schemas, and inter-module handoffs.

---

## 1. Core Types (`gpunity/types.py`)

All modules import from `gpunity.types`. No module defines its own data transfer objects.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

class OptimizationType(str, Enum):
    ATTENTION = "attention"
    COMPILATION = "compilation"
    MIXED_PRECISION = "mixed_precision"
    DATA_LOADING = "data_loading"
    PARALLELISM = "parallelism"
    MEMORY = "memory"
    KERNEL_FUSION = "kernel_fusion"
    COMMUNICATION = "communication"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class LoopDetectionMethod(str, Enum):
    USER_ANNOTATED = "user_annotated"
    HEURISTIC = "heuristic"
    NONE = "none"

@dataclass
class OperatorProfile:
    name: str
    gpu_time_ms: float
    cpu_time_ms: float
    pct_total: float
    call_count: int
    flops: Optional[int] = None

@dataclass
class MemoryTimelineEntry:
    step: int
    allocated_mb: float
    reserved_mb: float

@dataclass
class BackwardOpProfile:
    name: str
    time_ms: float
    pct_backward: float

@dataclass
class ProfileResult:
    # torch.profiler
    top_operators: list[OperatorProfile]
    gpu_utilization: float
    chrome_trace_path: Optional[Path] = None

    # Memory
    peak_memory_mb: float = 0.0
    memory_timeline: list[MemoryTimelineEntry] = field(default_factory=list)
    peak_allocation_stack: str = ""

    # Autograd
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    backward_ops: list[BackwardOpProfile] = field(default_factory=list)

    # DataLoader
    dataloader_throughput: float = 0.0
    dataloader_stall_time_ms: float = 0.0
    dataloader_bottleneck: bool = False

    # Loop detection metadata (D-001)
    loop_detection_method: LoopDetectionMethod = LoopDetectionMethod.NONE
    loop_detection_confidence: Optional[str] = None  # "high", "medium", "low", None

    # Artifact paths
    artifact_dir: Optional[Path] = None

    def summary(self) -> str:
        """Human-readable summary for the LLM agent."""
        ...

@dataclass
class OptimizationConfig:
    id: str                                    # "opt-001"
    name: str                                  # "Flash Attention + bf16"
    description: str                           # Human-readable explanation
    optimization_type: OptimizationType
    evidence: list[str]                        # ["sdpa = 58% GPU time", ...]
    code_changes: dict[str, str]               # {file_path: new_file_content}  (D-002)
    config_overrides: dict[str, object]        # {"precision": "bf16", "compile": True}
    estimated_speedup: float                   # 2.1
    estimated_memory_delta: float              # -0.35
    risk_level: RiskLevel
    dependencies: list[str] = field(default_factory=list)  # ["flash-attn>=2.5"]

@dataclass
class ControlRun:
    steps_completed: int
    wall_clock_seconds: float
    avg_step_time_ms: float
    peak_memory_mb: float
    throughput_samples_sec: float
    loss_values: list[float]
    gradient_norms: list[float]
    cost_estimate_usd: float

@dataclass
class ValidationResult:
    config_id: str
    config_name: str
    success: bool
    error: Optional[str] = None

    # Performance deltas (vs control)
    speedup_vs_control: float = 1.0
    memory_delta_vs_control: float = 0.0
    cost_delta_vs_control: float = 0.0

    # Raw metrics
    wall_clock_seconds: float = 0.0
    avg_step_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_samples_sec: float = 0.0
    loss_values: list[float] = field(default_factory=list)

    # Stability
    gradient_cosine_similarities: list[float] = field(default_factory=list)
    diverged: bool = False
    divergence_step: Optional[int] = None
    divergence_reason: str = ""

@dataclass
class RunConfig:
    """Merged CLI + YAML configuration."""
    repo_path: Path
    entry_point: str = "train.py"
    train_function: Optional[str] = None
    provider: str = "claude"
    model: Optional[str] = None
    gpu_type: str = "a100-80gb"
    profile_steps: int = 20
    warmup_steps: int = 5
    validation_steps: int = 50
    max_configs: int = 10
    top_k: int = 5
    divergence_threshold: float = 0.8
    gradient_check_interval: int = 5
    max_cost_per_sandbox: float = 5.0
    output_path: Path = Path("./gpunity_report.md")
    charts_mode: str = "ascii"
    verbose: bool = False
    dry_run: bool = False
    local: bool = False  # (D-003) use local runner instead of Modal
```

---

## 2. Module Interfaces

### 2.1 CLI (`gpunity/cli.py`)

```python
# Entry point: Click CLI
# Parses args, loads YAML config, merges into RunConfig, calls run_pipeline()

def run_pipeline(config: RunConfig) -> Path:
    """Execute the full pipeline. Returns path to the generated report."""
    ...
```

### 2.2 Config (`gpunity/config.py`)

```python
def load_config(cli_args: dict, yaml_path: Optional[Path] = None) -> RunConfig:
    """Merge CLI args with optional YAML config file. CLI args take precedence."""
    ...
```

### 2.3 Profiler (`gpunity/profiler/`)

```python
# profiler/wrapper.py
def create_profiling_wrapper(
    repo_path: Path,
    entry_point: str,
    train_function: Optional[str],
    warmup_steps: int,
    profile_steps: int,
) -> str:
    """Generate a Python wrapper script that instruments the user's training code.
    Returns the wrapper script content as a string."""
    ...

def detect_training_loop(source_code: str) -> Optional[dict]:
    """AST-based heuristic to find training loop boundaries.
    Returns {"function": name, "lineno": int} or None. (D-001)"""
    ...

# profiler/torch_profiler.py
def setup_torch_profiler(warmup: int, active: int, artifact_dir: Path) -> dict:
    """Returns profiler config dict for use in wrapper script."""
    ...

def extract_operator_profiles(trace_path: Path) -> list[OperatorProfile]:
    """Parse Chrome trace or key_averages to extract operator profiles."""
    ...

# profiler/memory.py
def extract_memory_profile(snapshot_path: Path) -> tuple[float, list[MemoryTimelineEntry], str]:
    """Returns (peak_mb, timeline, peak_stack)."""
    ...

# profiler/autograd.py
def extract_autograd_profile(profile_path: Path) -> tuple[float, float, list[BackwardOpProfile]]:
    """Returns (forward_ms, backward_ms, backward_ops)."""
    ...

# profiler/dataloader.py
def extract_dataloader_stats(stats_path: Path) -> tuple[float, float, bool]:
    """Returns (throughput, stall_time_ms, is_bottleneck)."""
    ...

# profiler/aggregate.py
def aggregate_profile(artifact_dir: Path) -> ProfileResult:
    """Read all profiler artifacts from artifact_dir and build ProfileResult."""
    ...
```

### 2.4 Sandbox (`gpunity/sandbox/`)

```python
# sandbox/modal_runner.py
async def run_in_modal(
    repo_path: Path,
    script_content: str,
    gpu_type: str,
    timeout_seconds: int,
    cost_ceiling_usd: float,
    extra_deps: list[str] | None = None,
) -> Path:
    """Run script in Modal sandbox. Returns local path to downloaded artifacts dir."""
    ...

# sandbox/local_runner.py  (D-003)
async def run_locally(
    repo_path: Path,
    script_content: str,
    timeout_seconds: int,
) -> Path:
    """Run script in a local subprocess. Returns path to artifacts dir."""
    ...

# sandbox/image_builder.py
def build_image_spec(repo_path: Path, extra_deps: list[str] | None = None) -> dict:
    """Parse repo deps and return Modal image specification."""
    ...

# sandbox/artifact_io.py
def upload_artifacts(local_dir: Path, sandbox_path: Path) -> None: ...
def download_artifacts(sandbox_path: Path, local_dir: Path) -> None: ...
```

### 2.5 Agent (`gpunity/agent/`)

```python
# agent/loop.py
async def run_agent_loop(
    profile: ProfileResult,
    repo_path: Path,
    config: RunConfig,
) -> list[OptimizationConfig]:
    """Run the LLM agent loop. Returns ranked, diversity-filtered configs."""
    ...

# agent/tools.py
def get_agent_tools(profile: ProfileResult, repo_path: Path) -> list[dict]:
    """Return tool definitions for the agent (read_profile, read_file, etc.)."""
    ...

def execute_tool(tool_name: str, tool_input: dict, context: dict) -> str:
    """Execute an agent tool call and return the result as a string."""
    ...

# agent/prompts.py
def get_system_prompt(profile_summary: str) -> str:
    """Return the system prompt for the optimization agent."""
    ...

# agent/diversity.py
def select_diverse_configs(
    configs: list[OptimizationConfig],
    top_k: int,
    diversity_threshold: float = 0.5,
) -> list[OptimizationConfig]:
    """Select top_k configs with diversity penalty for same optimization_type."""
    ...

# agent/providers/base.py
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.3,
    ) -> dict:
        """Returns {"content": str, "tool_calls": list[dict] | None}."""
        ...

# agent/providers/claude.py
class ClaudeProvider:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"): ...
    async def complete(self, messages, tools=None, temperature=0.3) -> dict: ...

# agent/providers/openai_provider.py
class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o"): ...
    async def complete(self, messages, tools=None, temperature=0.3) -> dict: ...
```

### 2.6 Validator (`gpunity/validator/`)

```python
# validator/runner.py
async def run_validation(
    control_repo: Path,
    configs: list[OptimizationConfig],
    run_config: RunConfig,
) -> tuple[ControlRun, list[ValidationResult]]:
    """Run control + all config variants. Returns control run and per-config results."""
    ...

# validator/patcher.py
def apply_config(repo_path: Path, config: OptimizationConfig) -> Path:
    """Apply code_changes and config_overrides to a copy of the repo.
    Returns path to the patched repo copy."""
    ...

# validator/gradient_tracker.py
def create_gradient_tracking_wrapper(
    entry_point: str,
    steps: int,
    gradient_check_interval: int,
    artifact_dir: Path,
) -> str:
    """Generate wrapper script that tracks gradients + loss + metrics during validation."""
    ...

def load_gradient_checkpoints(artifact_dir: Path) -> list[dict]:
    """Load saved gradient tensors and loss values from artifacts."""
    ...

# validator/divergence.py
def check_divergence(
    control_grads: list[dict],
    variant_grads: list[dict],
    threshold: float = 0.8,
    consecutive_failures: int = 3,
    loss_ratio_limit: float = 2.0,
) -> tuple[bool, Optional[int], str]:
    """Returns (diverged, divergence_step, reason)."""
    ...

# validator/metrics.py
def compute_validation_metrics(
    artifact_dir: Path,
    control: ControlRun,
) -> ValidationResult:
    """Compute performance deltas and stability metrics from validation artifacts."""
    ...
```

### 2.7 Reporter (`gpunity/reporter/`)

```python
# reporter/markdown.py
def generate_report(
    run_config: RunConfig,
    profile: ProfileResult,
    configs: list[OptimizationConfig],
    control: ControlRun,
    results: list[ValidationResult],
    output_path: Path,
) -> Path:
    """Generate the Markdown report. Returns the output path."""
    ...

# reporter/charts.py
def render_operator_breakdown(operators: list[OperatorProfile], mode: str = "ascii") -> str:
    """Render operator breakdown as ASCII table or base64 PNG."""
    ...

def render_loss_comparison(
    control_losses: list[float],
    variant_losses: dict[str, list[float]],
    mode: str = "ascii",
) -> str:
    """Render loss curve comparison."""
    ...

def render_cosine_sim_chart(
    cosine_sims: dict[str, list[float]],
    threshold: float,
    mode: str = "ascii",
) -> str:
    """Render gradient cosine similarity over steps."""
    ...

def render_summary_table(
    control: ControlRun,
    results: list[ValidationResult],
) -> str:
    """Render the validation summary comparison table."""
    ...
```

### 2.8 Utils (`gpunity/utils/`)

```python
# utils/logging.py
def get_logger(name: str, verbose: bool = False) -> logging.Logger: ...

# utils/cost.py
GPU_COSTS_PER_HOUR: dict[str, float]  # {"a100-80gb": 3.00, ...}
def estimate_cost(gpu_type: str, seconds: float) -> float: ...

# utils/repo.py
def scan_repo(repo_path: Path) -> dict:
    """Scan repo for entry point, requirements, structure."""
    ...

def detect_dependencies(repo_path: Path) -> list[str]:
    """Parse requirements.txt / pyproject.toml / setup.py."""
    ...
```

---

## 3. Artifact Directory Conventions

```
artifacts/
  profile/
    chrome_trace.json
    key_averages.txt
    memory_snapshot.json
    autograd_profile.json
    dataloader_stats.json
    profile_result.json          # Serialized ProfileResult
  configs/
    opt-001/
      config.json                # Serialized OptimizationConfig
      code_changes/              # Full file contents to write
    opt-002/
      ...
  validate/
    control/
      metrics.json
      loss.csv
      gradients/                 # step_005.pt, step_010.pt, ...
    opt-001/
      metrics.json
      loss.csv
      gradients/
    opt-002/
      ...
  report/
    gpunity_report.md
```

---

## 4. Phase Boundaries (Data Flow)

```
CLI (RunConfig)
  |
  v
Profiler (RunConfig) --> ProfileResult + artifacts/profile/
  |
  v
Agent (ProfileResult, repo_path, RunConfig) --> list[OptimizationConfig]
  |
  v
Validator (repo_path, list[OptimizationConfig], RunConfig) --> (ControlRun, list[ValidationResult])
  |
  v
Reporter (RunConfig, ProfileResult, configs, ControlRun, results) --> report.md
```

Each phase receives only the types defined above. No phase reaches into another phase's internals.
