"""GPUnity CLI entry point.

Provides the `gpunity` command with subcommands: run, profile, analyze, validate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

from gpunity.config import load_config
from gpunity.types import RunConfig


@click.group()
@click.version_option(version="0.1.0", prog_name="gpunity")
def main() -> None:
    """GPUnity: ML Training Optimization Pipeline.

    Profile, analyze, validate, and report on ML training optimizations.
    """
    pass


@main.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--entry", "entry_point", default="train.py", help="Training script entry point.")
@click.option("--train-function", default=None, help="Function containing the training loop.")
@click.option("--provider", default="claude", type=click.Choice(["claude", "openai", "custom"]),
              help="LLM provider.")
@click.option("--model", default=None, help="Model override for the provider.")
@click.option("--gpu", "gpu_type", default="a100-80gb",
              type=click.Choice(["a100-80gb", "a100-40gb", "h100", "a10g"]),
              help="GPU type for sandboxes.")
@click.option("--profile-steps", default=20, type=int, help="Steps to profile.")
@click.option("--warmup-steps", default=5, type=int, help="Warmup steps before profiling.")
@click.option("--validation-steps", default=50, type=int, help="Steps per validation run.")
@click.option("--max-configs", default=10, type=int, help="Total configs to generate.")
@click.option("--top-k", default=5, type=int, help="Configs to validate.")
@click.option("--divergence-threshold", default=0.8, type=float, help="Cosine sim threshold.")
@click.option("--max-cost", "max_cost_per_sandbox", default=5.0, type=float,
              help="Cost ceiling per sandbox (USD).")
@click.option("--output", "output_path", default="./gpunity_report.md",
              type=click.Path(), help="Report output path.")
@click.option("--charts", "charts_mode", default="ascii", type=click.Choice(["ascii", "png"]),
              help="Chart rendering mode.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
@click.option("--dry-run", is_flag=True, help="Run profile + analyze only, skip validation.")
@click.option("--local", is_flag=True, help="Use local runner instead of Modal.")
@click.option("--config-file", default=None, type=click.Path(exists=True),
              help="YAML configuration file.")
def run(repo_path: str, config_file: Optional[str], **kwargs: object) -> None:
    """Run the full GPUnity optimization pipeline.

    REPO_PATH is the path to the ML training repository.
    """
    cli_args = {"repo_path": repo_path, **kwargs}

    yaml_path = Path(config_file) if config_file else None
    config = load_config(cli_args, yaml_path)

    click.echo(f"GPUnity v0.1.0 -- Optimizing {config.repo_path}")
    click.echo(f"  Entry point: {config.entry_point}")
    click.echo(f"  Provider: {config.provider}")
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
@click.option("--local", is_flag=True, help="Use local runner.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def profile(repo_path: str, **kwargs: object) -> None:
    """Run profiling only (Phase 1).

    REPO_PATH is the path to the ML training repository.
    """
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)

    click.echo(f"GPUnity -- Profiling {config.repo_path}")
    result = asyncio.run(run_profile_phase(config))
    click.echo(f"\nProfile complete. Artifacts in: {result.artifact_dir}")
    click.echo(result.summary())


@main.command()
@click.argument("profile_dir", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the repo.")
@click.option("--provider", default="claude", help="LLM provider.")
@click.option("--model", default=None, help="Model override.")
@click.option("--max-configs", default=10, type=int, help="Total configs to generate.")
@click.option("--top-k", default=5, type=int, help="Configs to select.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def analyze(profile_dir: str, repo_path: str, **kwargs: object) -> None:
    """Run analysis only (Phase 2) from saved profile artifacts.

    PROFILE_DIR is the path to saved profile artifacts.
    """
    click.echo(f"GPUnity -- Analyzing profile from {profile_dir}")
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)
    configs = asyncio.run(run_analyze_phase(profile_dir, config))
    click.echo(f"\nGenerated {len(configs)} optimization configs.")


@main.command()
@click.argument("config_dir", type=click.Path(exists=True))
@click.option("--repo", "repo_path", required=True, type=click.Path(exists=True),
              help="Path to the repo.")
@click.option("--validation-steps", default=50, type=int, help="Steps per validation run.")
@click.option("--divergence-threshold", default=0.8, type=float, help="Cosine sim threshold.")
@click.option("--local", is_flag=True, help="Use local runner.")
@click.option("--verbose", is_flag=True, help="Show detailed logs.")
def validate(config_dir: str, repo_path: str, **kwargs: object) -> None:
    """Run validation only (Phase 3) from saved configs.

    CONFIG_DIR is the path to saved optimization configs.
    """
    click.echo(f"GPUnity -- Validating configs from {config_dir}")
    cli_args = {"repo_path": repo_path, **kwargs}
    config = load_config(cli_args)
    asyncio.run(run_validate_phase(config_dir, config))


async def run_pipeline(config: RunConfig) -> Path:
    """Execute the full GPUnity pipeline.

    Phase 1: Profile the training script
    Phase 2: Generate optimization configs via LLM agent
    Phase 3: Validate configs in parallel (unless dry_run)
    Phase 4: Generate report

    Args:
        config: The merged run configuration.

    Returns:
        Path to the generated report file.
    """
    from gpunity.utils.logging import get_logger

    logger = get_logger("gpunity.pipeline", config.verbose)

    # Phase 1: Profile
    logger.info("Phase 1: Profiling training script...")
    profile_result = await run_profile_phase(config)
    logger.info("Phase 1 complete.")

    # Phase 2: Analyze
    logger.info("Phase 2: Generating optimization configs...")
    from gpunity.agent.loop import run_agent_loop
    optimization_configs = await run_agent_loop(profile_result, Path(config.repo_path), config)
    logger.info(f"Phase 2 complete. Generated {len(optimization_configs)} configs.")

    # Phase 3: Validate (unless dry_run)
    control_run = None
    validation_results = []
    if not config.dry_run:
        logger.info("Phase 3: Validating configs...")
        from gpunity.validator.runner import run_validation
        control_run, validation_results = await run_validation(
            Path(config.repo_path), optimization_configs, config
        )
        logger.info(f"Phase 3 complete. {len(validation_results)} configs validated.")
    else:
        logger.info("Phase 3: Skipped (dry-run mode).")
        from gpunity.types import ControlRun
        control_run = ControlRun(
            steps_completed=0, wall_clock_seconds=0, avg_step_time_ms=0,
            peak_memory_mb=0, throughput_samples_sec=0, loss_values=[],
            gradient_norms=[], cost_estimate_usd=0,
        )

    # Phase 4: Report
    logger.info("Phase 4: Generating report...")
    from gpunity.reporter.markdown import generate_report
    output_path = Path(config.output_path)
    report_path = generate_report(
        run_config=config,
        profile=profile_result,
        configs=optimization_configs,
        control=control_run,
        results=validation_results,
        output_path=output_path,
    )
    logger.info(f"Phase 4 complete. Report at: {report_path}")

    return report_path


async def run_profile_phase(config: RunConfig):
    """Execute Phase 1: Profiling."""
    from gpunity.profiler.wrapper import create_profiling_wrapper
    from gpunity.profiler.aggregate import aggregate_profile

    repo_path = Path(config.repo_path)

    # Generate profiling wrapper script
    wrapper_script = create_profiling_wrapper(
        repo_path=repo_path,
        entry_point=config.entry_point,
        train_function=config.train_function,
        warmup_steps=config.warmup_steps,
        profile_steps=config.profile_steps,
    )

    # Run in sandbox
    if config.local:
        from gpunity.sandbox.local_runner import run_locally
        artifact_dir = await run_locally(
            repo_path=repo_path,
            script_content=wrapper_script,
            timeout_seconds=300,
        )
    else:
        from gpunity.sandbox.modal_runner import run_in_modal
        artifact_dir = await run_in_modal(
            repo_path=repo_path,
            script_content=wrapper_script,
            gpu_type=config.gpu_type,
            timeout_seconds=300,
            cost_ceiling_usd=config.max_cost_per_sandbox,
        )

    # Aggregate results
    profile_result = aggregate_profile(artifact_dir)
    return profile_result


async def run_analyze_phase(profile_dir: str, config: RunConfig):
    """Execute Phase 2: Analysis from saved profile."""
    from gpunity.profiler.aggregate import aggregate_profile
    from gpunity.agent.loop import run_agent_loop

    profile_result = aggregate_profile(Path(profile_dir))
    configs = await run_agent_loop(profile_result, Path(config.repo_path), config)
    return configs


async def run_validate_phase(config_dir: str, config: RunConfig):
    """Execute Phase 3: Validation from saved configs."""
    import json

    from gpunity.types import OptimizationConfig
    from gpunity.validator.runner import run_validation

    config_path = Path(config_dir)
    configs = []
    for cf in sorted(config_path.glob("*/config.json")):
        with open(cf) as f:
            configs.append(OptimizationConfig.from_dict(json.load(f)))

    control_run, results = await run_validation(Path(config.repo_path), configs, config)
    return control_run, results


if __name__ == "__main__":
    main()
