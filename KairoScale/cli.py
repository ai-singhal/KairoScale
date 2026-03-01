"""KairoScale CLI entry point.

Provides the `KairoScale` command with subcommands: run, profile, analyze, validate.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from KairoScale.config import load_config
from KairoScale.types import RunConfig


@click.group()
@click.version_option(version="0.1.0", prog_name="KairoScale")
def main() -> None:
    """KairoScale: ML Training Optimization Pipeline.

    Profile, analyze, validate, and report on ML training optimizations.
    """
    pass


def _export_configs(configs, output_dir: Path) -> Path:
    """Persist generated configs to disk for reproducible application."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        cfg_path = output_dir / f"{cfg.id}.json"
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return output_dir


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--entry", "entry_point", default="train.py", help="Training script entry point.")
@click.option("--train-function", default=None, help="Function containing the training loop.")
@click.option("--provider", default="claude",
              type=click.Choice(["claude", "openai", "heuristic", "custom", "modal"]),
              help="LLM provider.")
@click.option("--model", default=None, help="Model override for the provider.")
@click.option("--mode", default="auto", type=click.Choice(["auto", "train", "infer"]),
              help="Workload mode.")
@click.option("--baseline-policy", default="required",
              type=click.Choice(["required", "minimal"]),
              help="Native baseline ladder policy.")
@click.option("--compare-against-native/--no-compare-against-native", default=True,
              help="Compare winner against best native baseline.")
@click.option("--hardware-profile", default="auto", type=click.Choice(["auto", "manual"]),
              help="Hardware profile mode.")
@click.option("--objective-profile", default="balanced",
              type=click.Choice(["balanced", "latency", "cost", "throughput"]),
              help="Objective profile for winner selection.")
@click.option("--ablation-top-k", default=3, type=int,
              help="Attach ablation contribution notes for top-k scored configs.")
@click.option("--gpu", "gpu_type", default="a100-80gb",
              type=click.Choice(["a100-80gb", "a100-40gb", "h100", "a10g"]),
              help="GPU type for sandboxes.")
@click.option("--gpu-count", default=1, type=int,
              help="Target number of GPUs (set >1 for cluster-oriented baseline policy).")
@click.option("--profile-steps", default=20, type=int, help="Steps to profile.")
@click.option("--warmup-steps", default=5, type=int, help="Warmup steps before profiling.")
@click.option("--validation-steps", default=50, type=int, help="Steps per validation run.")
@click.option("--max-configs", default=10, type=int, help="Total configs to generate.")
@click.option("--top-k", default=5, type=int, help="Configs to validate.")
@click.option("--divergence-threshold", default=0.8, type=float, help="Cosine sim threshold.")
@click.option("--logits-tolerance", default=1e-3, type=float,
              help="Max allowed absolute logit signature delta vs control.")
@click.option("--max-cost", "max_cost_per_sandbox", default=5.0, type=float,
              help="Cost ceiling per sandbox (USD).")
@click.option("--validation-seed", default=1337, type=int,
              help="Seed used for deterministic validation replay.")
@click.option("--deterministic-validation/--no-deterministic-validation", default=True,
              help="Enable deterministic validation seeding and dataloader patching.")
@click.option("--validation-strategy",
              default="parallel_all",
              type=click.Choice(["parallel_all", "staged"]),
              help="Validation fanout strategy across sandboxes.")
@click.option("--staged-top-k", default=2, type=int,
              help="For staged strategy: number of configs promoted to full validation.")
@click.option("--output", "output_path", default="./KairoScale_report.md",
              type=click.Path(), help="Report output path.")
@click.option("--charts", "charts_mode", default="ascii", type=click.Choice(["ascii", "png"]),
              help="Chart rendering mode.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
@click.option("--dry-run", is_flag=True, help="Run profile + analyze only, skip validation.")
@click.option("--local", is_flag=True, help="Use local runner instead of Modal.")
@click.option("--python-bin", default=None, type=click.Path(exists=True),
              help="Python binary for local wrapper execution.")
@click.option("--gpu-selection", default="auto", type=click.Choice(["auto", "fixed"]),
              help="GPU selection mode. 'auto' tests cheaper GPUs when compute-bound.")
@click.option("--gpu-aggressiveness", default="moderate",
              type=click.Choice(["none", "conservative", "moderate", "aggressive"]),
              help="How aggressively to test cheaper GPUs.")
@click.option("--config-file", default=None, type=click.Path(exists=True),
              help="YAML configuration file.")
def run(repo_path: str, config_file: Optional[str], **kwargs: object) -> None:
    """Run the full KairoScale optimization pipeline.

    REPO_PATH is the path to the ML training repository.
    """
    cli_args = {"repo_path": repo_path, **kwargs}

    yaml_path = Path(config_file) if config_file else None
    config = load_config(cli_args, yaml_path)

    click.echo(f"KairoScale v0.1.0 -- Optimizing {config.repo_path}")
    click.echo(f"  Entry point: {config.entry_point}")
    click.echo(f"  Provider: {config.provider}")
    click.echo(f"  Objective: {config.objective_profile}")
    click.echo(f"  Compare native: {config.compare_against_native}")
    click.echo(f"  Mode: {'local' if config.local else 'modal'}")
    click.echo()

    report_path = asyncio.run(run_pipeline(config))
    click.echo(f"\nReport written to: {report_path}")


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--entry", "entry_point", default="train.py", help="Training script entry point.")
@click.option("--train-function", default=None, help="Function containing the training loop.")
@click.option("--profile-steps", default=20, type=int, help="Steps to profile.")
@click.option("--warmup-steps", default=5, type=int, help="Warmup steps before profiling.")
@click.option("--gpu", "gpu_type", default="a100-80gb", help="GPU type.")
@click.option("--gpu-count", default=1, type=int, help="Target number of GPUs.")
@click.option("--local", is_flag=True, help="Use local runner.")
@click.option("--python-bin", default=None, type=click.Path(exists=True),
              help="Python binary for local wrapper execution.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def profile(repo_path: str, **kwargs: object) -> None:
    """Run profiling only (Phase 1).

    REPO_PATH is the path to the ML training repository.
    """
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)

    click.echo(f"KairoScale -- Profiling {config.repo_path}")
    result = asyncio.run(run_profile_phase(config))
    click.echo(f"\nProfile complete. Artifacts in: {result.artifact_dir}")
    click.echo(result.summary())


@main.command()
@click.argument("profile_dir", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the repo.")
@click.option("--provider", default="claude",
              type=click.Choice(["claude", "openai", "heuristic", "custom", "modal"]),
              help="LLM provider.")
@click.option("--model", default=None, help="Model override.")
@click.option("--max-configs", default=10, type=int, help="Total configs to generate.")
@click.option("--top-k", default=5, type=int, help="Configs to select.")
@click.option("--gpu-count", default=1, type=int, help="Target number of GPUs.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def analyze(profile_dir: str, repo_path: str, **kwargs: object) -> None:
    """Run analysis only (Phase 2) from saved profile artifacts.

    PROFILE_DIR is the path to saved profile artifacts.
    """
    click.echo(f"KairoScale -- Analyzing profile from {profile_dir}")
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)
    configs = asyncio.run(run_analyze_phase(profile_dir, config))
    click.echo(f"\nGenerated {len(configs)} optimization configs.")


@main.command()
@click.argument("config_dir", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the repo.")
@click.option("--mode", default="auto", type=click.Choice(["auto", "train", "infer"]),
              help="Workload mode.")
@click.option("--baseline-policy", default="required",
              type=click.Choice(["required", "minimal"]),
              help="Native baseline ladder policy.")
@click.option("--compare-against-native/--no-compare-against-native", default=True,
              help="Compare winner against best native baseline.")
@click.option("--hardware-profile", default="auto", type=click.Choice(["auto", "manual"]),
              help="Hardware profile mode.")
@click.option("--objective-profile", default="balanced",
              type=click.Choice(["balanced", "latency", "cost", "throughput"]),
              help="Objective profile for winner selection.")
@click.option("--ablation-top-k", default=3, type=int,
              help="Attach ablation contribution notes for top-k scored configs.")
@click.option("--validation-steps", default=50, type=int, help="Steps per validation run.")
@click.option("--divergence-threshold", default=0.8, type=float, help="Cosine sim threshold.")
@click.option("--logits-tolerance", default=1e-3, type=float,
              help="Max allowed absolute logit signature delta vs control.")
@click.option("--validation-seed", default=1337, type=int,
              help="Seed used for deterministic validation replay.")
@click.option("--deterministic-validation/--no-deterministic-validation", default=True,
              help="Enable deterministic validation seeding and dataloader patching.")
@click.option("--validation-strategy",
              default="parallel_all",
              type=click.Choice(["parallel_all", "staged"]),
              help="Validation fanout strategy across sandboxes.")
@click.option("--staged-top-k", default=2, type=int,
              help="For staged strategy: number of configs promoted to full validation.")
@click.option("--gpu-count", default=1, type=int, help="Target number of GPUs.")
@click.option("--local", is_flag=True, help="Use local runner.")
@click.option("--python-bin", default=None, type=click.Path(exists=True),
              help="Python binary for local wrapper execution.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def validate(config_dir: str, repo_path: str, **kwargs: object) -> None:
    """Run validation only (Phase 3) from saved configs.

    CONFIG_DIR is the path to saved optimization configs.
    """
    click.echo(f"KairoScale -- Validating configs from {config_dir}")
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)
    asyncio.run(run_validate_phase(config_dir, config))


@main.command("apply")
@click.argument("config_json", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the repo where changes will be applied.")
def apply_config_cmd(config_json: str, repo_path: str) -> None:
    """Apply a saved optimization config JSON directly to a repository."""
    from KairoScale.types import OptimizationConfig
    from KairoScale.validator.patcher import apply_config_in_place

    config_path = Path(config_json)
    with open(config_path) as f:
        config = OptimizationConfig.from_dict(json.load(f))

    written = apply_config_in_place(Path(repo_path), config)
    click.echo(f"Applied config {config.id} ({config.name}) to {repo_path}")
    for path in written:
        click.echo(f"  wrote: {path}")


@main.command("deploy")
@click.argument("config_json", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the ML training repository.")
@click.option("--gpu", "gpu_type", default="a100-80gb",
              type=click.Choice(["a100-80gb", "a100-40gb", "h100", "a10g"]),
              help="GPU type for Modal sandbox.")
@click.option("--entry", "entry_point", default="train.py",
              help="Training script entry point.")
@click.option("--steps", "train_steps", default=None, type=int,
              help="Number of training steps (passed as TRAIN_STEPS env var).")
@click.option("--local", is_flag=True, help="Run locally instead of on Modal.")
@click.option("--python-bin", default=None, type=click.Path(exists=True),
              help="Python binary for local execution.")
@click.option("--timeout", "timeout_seconds", default=3600, type=int,
              help="Maximum execution time in seconds.")
def deploy(
    config_json: str,
    repo_path: str,
    gpu_type: str,
    entry_point: str,
    train_steps: Optional[int],
    local: bool,
    python_bin: Optional[str],
    timeout_seconds: int,
) -> None:
    """Deploy a winning optimization config for full training on Modal.

    Applies the optimization config to the repo and runs full training
    in a GPU sandbox. This is for production training runs, not profiling.

    CONFIG_JSON is the path to a saved optimization config JSON file.
    """
    from KairoScale.types import OptimizationConfig
    from KairoScale.validator.patcher import apply_config

    config_path = Path(config_json)
    with open(config_path) as f:
        opt_config = OptimizationConfig.from_dict(json.load(f))

    click.echo(f"KairoScale Deploy -- {opt_config.name}")
    click.echo(f"  Config: {opt_config.id}")
    click.echo(f"  GPU: {gpu_type}")
    click.echo(f"  Repo: {repo_path}")
    click.echo()

    # Apply config to a temp copy
    patched_repo = apply_config(Path(repo_path), opt_config)
    click.echo(f"Applied config to: {patched_repo}")

    # Generate a simple training wrapper (not the profiling wrapper)
    train_steps_env = f'os.environ["TRAIN_STEPS"] = "{train_steps}"' if train_steps else ""
    wrapper_script = f'''#!/usr/bin/env python3
"""KairoScale deploy wrapper -- runs optimized training."""

import importlib.util
import os
import sys
import time

repo_dir = os.environ.get("REPO_DIR", os.getcwd())
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)

{train_steps_env}

print("[KairoScale deploy] Starting optimized training...")
print(f"[KairoScale deploy] Entry point: {entry_point}")
start = time.time()

# Load and run the entry point
spec = importlib.util.spec_from_file_location(
    "__train_module__",
    os.path.join(repo_dir, "{entry_point}"),
)
if spec is None or spec.loader is None:
    print(f"[KairoScale deploy] ERROR: Cannot load {entry_point}")
    sys.exit(1)

module = importlib.util.module_from_spec(spec)
sys.modules["__train_module__"] = module

try:
    spec.loader.exec_module(module)
    # Try common entry functions if nothing ran
    for candidate in ("train", "main", "run"):
        fn = getattr(module, candidate, None)
        if callable(fn):
            fn()
            break
except Exception as e:
    print(f"[KairoScale deploy] ERROR: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

elapsed = time.time() - start
print(f"[KairoScale deploy] Training complete in {{elapsed:.1f}}s")
'''

    click.echo("Launching training...")
    artifact_dir = asyncio.run(
        _run_deploy(
            patched_repo=patched_repo,
            wrapper_script=wrapper_script,
            gpu_type=gpu_type,
            local=local,
            python_bin=python_bin,
            timeout_seconds=timeout_seconds,
        )
    )
    click.echo(f"\nDeploy complete. Artifacts in: {artifact_dir}")


async def _run_deploy(
    patched_repo: Path,
    wrapper_script: str,
    gpu_type: str,
    local: bool,
    python_bin: Optional[str],
    timeout_seconds: int,
) -> Path:
    """Execute the deploy run in sandbox."""
    if local:
        from KairoScale.sandbox.local_runner import run_locally
        return await run_locally(
            repo_path=patched_repo,
            script_content=wrapper_script,
            timeout_seconds=timeout_seconds,
            python_bin=python_bin,
        )
    else:
        from KairoScale.sandbox.modal_runner import run_in_modal
        return await run_in_modal(
            repo_path=patched_repo,
            script_content=wrapper_script,
            gpu_type=gpu_type,
            timeout_seconds=timeout_seconds,
            stream_logs=True,
        )


async def run_pipeline(config: RunConfig) -> Path:
    """Execute the full KairoScale pipeline.

    Phase 1: Profile the training script
    Phase 2: Generate optimization configs via LLM agent
    Phase 3: Validate configs in parallel (unless dry_run)
    Phase 4: Generate report

    Args:
        config: The merged run configuration.

    Returns:
        Path to the generated report file.
    """
    from KairoScale.utils.logging import get_logger

    logger = get_logger("KairoScale.pipeline", config.verbose)

    # Phase 1: Profile
    logger.info("Phase 1: Profiling training script...")
    profile_result = await run_profile_phase(config)
    logger.info("Phase 1 complete.")

    from KairoScale.hardware.profile import detect_hardware_profile, resolve_workload_mode

    hardware = detect_hardware_profile(config)
    mode = resolve_workload_mode(config.mode, profile_result)
    logger.info(
        "Resolved mode=%s, hardware=%s (%s)",
        mode,
        hardware.gpu_name,
        hardware.detection_source,
    )

    # Phase 1.5: Bottleneck diagnosis
    diagnosis = None
    if config.gpu_selection == "auto":
        from KairoScale.diagnosis.bottleneck import diagnose_bottleneck
        diagnosis = diagnose_bottleneck(
            profile_result, hardware, config.gpu_aggressiveness
        )
        logger.info(
            "Bottleneck diagnosis: %s (confidence: %s)",
            diagnosis.primary.value,
            diagnosis.confidence,
        )
        for ev in diagnosis.evidence:
            logger.info("  Evidence: %s", ev)

    # Phase 2: Analyze
    logger.info("Phase 2: Generating optimization configs...")
    from KairoScale.agent.loop import run_agent_loop
    optimization_configs = await run_agent_loop(
        profile_result,
        Path(config.repo_path),
        config,
        hardware_profile=hardware,
        mode=mode,
        diagnosis=diagnosis,
    )
    logger.info(f"Phase 2 complete. Generated {len(optimization_configs)} configs.")
    config_export_dir = _export_configs(
        optimization_configs,
        Path(config.output_path).parent / "KairoScale_configs",
    )

    # Phase 3: Validate (unless dry_run)
    control_run = None
    validation_results = []
    run_summary = None
    if not config.dry_run:
        logger.info("Phase 3: Validating configs...")
        from KairoScale.agent.combo import generate_combo_configs
        from KairoScale.validator.runner import _run_validation_batch, run_validation
        control_run, validation_results, run_summary = await run_validation(
            Path(config.repo_path),
            optimization_configs,
            config,
            profile=profile_result,
            hardware_profile=hardware,
            mode=mode,
            diagnosis=diagnosis,
        )
        logger.info(f"Phase 3 complete. {len(validation_results)} configs validated.")

        logger.info("Phase 3.5: Generating combo configs...")
        result_by_id = {result.config_id: result for result in validation_results}
        config_result_pairs = [
            (cfg, result_by_id[cfg.id])
            for cfg in optimization_configs
            if cfg.id in result_by_id
        ]
        combo_configs = generate_combo_configs(config_result_pairs, max_combos=3)
        if combo_configs:
            _combo_control, combo_results = await _run_validation_batch(
                control_repo=Path(config.repo_path),
                configs=combo_configs,
                run_config=config,
                mode=mode,
                steps=config.validation_steps,
                batch_label="combo",
                existing_control=control_run,
            )
            optimization_configs.extend(combo_configs)
            validation_results.extend(combo_results)
            _export_configs(combo_configs, config_export_dir)
            logger.info(
                f"Phase 3.5 complete. Generated and validated {len(combo_results)} combos."
            )
        else:
            logger.info("Phase 3.5 complete. No compatible combos generated.")
    else:
        logger.info("Phase 3: Skipped (dry-run mode).")
        from KairoScale.types import ControlRun, RunSummary
        control_run = ControlRun(
            steps_completed=0, wall_clock_seconds=0, avg_step_time_ms=0,
            peak_memory_mb=0, throughput_samples_sec=0, loss_values=[],
            gradient_norms=[], cost_estimate_usd=0,
        )
        run_summary = RunSummary(objective_profile=config.objective_profile, confidence="low")

    # Phase 4: Report
    logger.info("Phase 4: Generating report...")
    from KairoScale.reporter.markdown import generate_report
    output_path = Path(config.output_path)
    report_path = generate_report(
        run_config=config,
        profile=profile_result,
        configs=optimization_configs,
        control=control_run,
        results=validation_results,
        output_path=output_path,
        config_export_dir=config_export_dir,
        hardware_profile=hardware,
        run_summary=run_summary,
        mode=mode,
    )
    logger.info(f"Phase 4 complete. Report at: {report_path}")

    return report_path


async def run_profile_phase(config: RunConfig):
    """Execute Phase 1: Profiling."""
    from KairoScale.profiler.wrapper import create_profiling_wrapper
    from KairoScale.profiler.aggregate import aggregate_profile

    repo_path = Path(config.repo_path)

    # Generate profiling wrapper script
    wrapper_script = create_profiling_wrapper(
        repo_path=repo_path,
        entry_point=config.entry_point,
        train_function=config.train_function,
        warmup_steps=config.warmup_steps,
        profile_steps=config.profile_steps,
    )

    # Scale timeout with requested profiling window to support production runs.
    total_profile_steps = config.warmup_steps + config.profile_steps
    profile_timeout_seconds = max(
        300,
        min(3600, 120 + total_profile_steps * 12),
    )

    # Run in sandbox
    if config.local:
        from KairoScale.sandbox.local_runner import run_locally
        artifact_dir = await run_locally(
            repo_path=repo_path,
            script_content=wrapper_script,
            timeout_seconds=profile_timeout_seconds,
            python_bin=config.python_bin,
        )
    else:
        from KairoScale.sandbox.modal_runner import run_in_modal
        artifact_dir = await run_in_modal(
            repo_path=repo_path,
            script_content=wrapper_script,
            gpu_type=config.gpu_type,
            timeout_seconds=profile_timeout_seconds,
            cost_ceiling_usd=config.max_cost_per_sandbox,
        )

    # Aggregate results
    profile_result = aggregate_profile(artifact_dir)
    return profile_result


async def run_analyze_phase(profile_dir: str, config: RunConfig):
    """Execute Phase 2: Analysis from saved profile."""
    from KairoScale.profiler.aggregate import aggregate_profile
    from KairoScale.hardware.profile import detect_hardware_profile, resolve_workload_mode
    from KairoScale.agent.loop import run_agent_loop

    profile_result = aggregate_profile(Path(profile_dir))
    hardware = detect_hardware_profile(config)
    mode = resolve_workload_mode(config.mode, profile_result)
    configs = await run_agent_loop(
        profile_result,
        Path(config.repo_path),
        config,
        hardware_profile=hardware,
        mode=mode,
    )
    return configs


async def run_validate_phase(config_dir: str, config: RunConfig):
    """Execute Phase 3: Validation from saved configs."""
    import json

    from KairoScale.hardware.profile import detect_hardware_profile, resolve_workload_mode
    from KairoScale.types import OptimizationConfig, ProfileResult
    from KairoScale.validator.runner import run_validation

    config_path = Path(config_dir)
    configs = []
    for cf in sorted(config_path.glob("*.json")):
        with open(cf) as f:
            configs.append(OptimizationConfig.from_dict(json.load(f)))

    hardware = detect_hardware_profile(config)
    mode = resolve_workload_mode(config.mode, None)
    control_run, results, run_summary = await run_validation(
        Path(config.repo_path),
        configs,
        config,
        profile=ProfileResult(),
        hardware_profile=hardware,
        mode=mode,
    )
    return control_run, results, run_summary


if __name__ == "__main__":
    main()
