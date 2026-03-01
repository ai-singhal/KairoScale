"""Streamlit command center for GPUnity optimization runs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure the project root is importable even when invoked via
# `streamlit run gpunity/ui/app.py` without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from gpunity.config import load_config
from gpunity.types import ControlRun, OptimizationConfig, RunSummary, ValidationResult


@dataclass
class DashboardResult:
    report_path: Path
    config_export_dir: Path
    control_run: ControlRun
    validation_results: list[ValidationResult]
    run_summary: RunSummary


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from synchronous Streamlit context.

    Streamlit may already have a running event loop, so asyncio.run()
    will raise RuntimeError.  Fall back to a fresh loop in that case.
    """
    try:
        asyncio.get_running_loop()
        # Already inside a loop — create a new one on a thread-safe basis.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError:
        # No running loop — safe to use asyncio.run().
        return asyncio.run(coro)


def _export_configs(configs: list[OptimizationConfig], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        cfg_path = output_dir / f"{cfg.id}.json"
        cfg_path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return output_dir


async def _execute_pipeline_phased(config: Any, status_container) -> DashboardResult:
    from gpunity.cli import run_profile_phase
    from gpunity.diagnosis.bottleneck import diagnose_bottleneck
    from gpunity.hardware.profile import detect_hardware_profile, resolve_workload_mode
    from gpunity.reporter.markdown import generate_report
    from gpunity.validator.runner import run_validation

    sandbox_mode = "Modal Sandbox" if not config.local else "Local Sandbox"
    provider_label = config.provider.capitalize()

    status_container.info(f"Phase 1: Profiling training script in {sandbox_mode}...")
    profile_result = await run_profile_phase(config)
    gpu_util = getattr(profile_result, "gpu_utilization", None)
    peak_mem = getattr(profile_result, "peak_memory_mb", None)
    phase1_msg = "Phase 1 complete."
    if gpu_util is not None:
        phase1_msg += f" GPU util: {gpu_util:.0f}%"
    if peak_mem is not None:
        phase1_msg += f", Peak VRAM: {peak_mem:.0f} MB"
    status_container.info(phase1_msg)

    hardware = detect_hardware_profile(config)
    mode = resolve_workload_mode(config.mode, profile_result)

    diagnosis = None
    if config.gpu_selection == "auto":
        diagnosis = diagnose_bottleneck(profile_result, hardware, config.gpu_aggressiveness)
        if diagnosis:
            status_container.info(
                f"Bottleneck: {diagnosis.primary.value} (confidence: {diagnosis.confidence})"
            )

    from gpunity.agent.loop import run_agent_loop

    status_container.info(f"Phase 2: {provider_label} LLM analyzing profile and proposing optimizations...")
    optimization_configs = await run_agent_loop(
        profile_result,
        Path(config.repo_path),
        config,
        hardware_profile=hardware,
        mode=mode,
        diagnosis=diagnosis,
    )
    status_container.info(f"Phase 2 complete. Generated {len(optimization_configs)} optimization configs.")

    config_export_dir = _export_configs(
        optimization_configs,
        Path(config.output_path).parent / "gpunity_configs",
    )

    status_container.info(
        f"Phase 3: Validating {len(optimization_configs)} configs + control in parallel {sandbox_mode}s..."
    )
    control_run, validation_results, run_summary = await run_validation(
        Path(config.repo_path),
        optimization_configs,
        config,
        profile=profile_result,
        hardware_profile=hardware,
        mode=mode,
        diagnosis=diagnosis,
    )
    n_diverged = sum(1 for r in validation_results if r.diverged)
    status_container.info(
        f"Phase 3 complete. {len(validation_results)} validated, {n_diverged} diverged."
    )

    status_container.info("Phase 4: Generating report...")
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

    return DashboardResult(
        report_path=report_path,
        config_export_dir=config_export_dir,
        control_run=control_run,
        validation_results=validation_results,
        run_summary=run_summary,
    )


def _to_points(
    control: ControlRun,
    results: list[ValidationResult],
    selected_ids: set[str],
    best_id: str | None,
    cost_cap_ratio: float,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    baseline_time = max(control.wall_clock_seconds, 0.0)
    baseline_cost = max(control.cost_estimate_usd, 0.0)

    points.append(
        {
            "id": "control",
            "name": "Control",
            "time": baseline_time,
            "cost": baseline_cost,
            "status": "control",
            "diverged": "no",
            "success": True,
            "eligible": True,
            "cost_delta": 0.0,
            "speedup": 1.0,
            "objective_score": 0.0,
        }
    )

    for result in results:
        if baseline_cost > 0:
            abs_cost = baseline_cost * (1.0 + result.cost_delta_vs_control)
        else:
            abs_cost = max(0.0, result.cost_delta_vs_control)

        time_value = result.wall_clock_seconds
        if time_value <= 0 and baseline_time > 0 and result.speedup_vs_control > 0:
            time_value = baseline_time / result.speedup_vs_control

        time_value = max(0.0, time_value)
        abs_cost = max(0.0, abs_cost)

        eligible = (
            result.success
            and not result.diverged
            and abs_cost <= baseline_cost * (1.0 + cost_cap_ratio)
            and time_value > 0
        )

        status = "candidate"
        if result.diverged:
            status = "diverged"
        elif result.config_id == best_id:
            status = "best"
        elif result.config_id in selected_ids:
            status = "alternative"

        points.append(
            {
                "id": result.config_id,
                "name": result.config_name,
                "time": time_value,
                "cost": abs_cost,
                "status": status,
                "diverged": "yes" if result.diverged else "no",
                "success": result.success,
                "eligible": eligible,
                "cost_delta": result.cost_delta_vs_control,
                "speedup": result.speedup_vs_control,
                "objective_score": result.objective_score if result.objective_score is not None else 0.0,
            }
        )

    return points


def _pick_recommendations(
    points: list[dict[str, Any]],
    max_alternatives: int = 2,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible = [
        p
        for p in points
        if p["id"] != "control" and p["eligible"] and p["success"] and p["diverged"] == "no"
    ]
    if not eligible:
        return None, []

    ranked = sorted(
        eligible,
        key=lambda p: (p["time"], p["cost"], -((p.get("speedup") or 0.0))),
    )
    best = ranked[0]
    alternatives = ranked[1 : 1 + max(0, max_alternatives)]
    return best, alternatives


def _pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        p
        for p in points
        if p["id"] != "control" and p["success"] and p["diverged"] == "no" and p["time"] > 0
    ]
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda p: (p["time"], p["cost"]))
    frontier: list[dict[str, Any]] = []
    best_cost = float("inf")
    for point in candidates:
        if point["cost"] < best_cost:
            frontier.append(point)
            best_cost = point["cost"]
    return frontier


def _render_chart(points: list[dict[str, Any]], frontier: list[dict[str, Any]], cap_cost: float) -> None:
    # Vega-Lite spec with explicit data per layer to avoid data-inheritance issues.
    scatter_layer: dict[str, Any] = {
        "data": {"values": points},
        "mark": {"type": "point", "filled": True, "size": 180},
        "encoding": {
            "x": {"field": "time", "type": "quantitative", "title": "Wall Clock Time (s)"},
            "y": {"field": "cost", "type": "quantitative", "title": "Estimated Cost (USD)"},
            "color": {
                "field": "status",
                "type": "nominal",
                "title": "Status",
                "scale": {
                    "domain": ["control", "best", "alternative", "candidate", "diverged"],
                    "range": ["#636363", "#2ca02c", "#1f77b4", "#aec7e8", "#d62728"],
                },
            },
            "shape": {
                "field": "diverged",
                "type": "nominal",
                "scale": {"domain": ["no", "yes"], "range": ["circle", "triangle-up"]},
            },
            "tooltip": [
                {"field": "name", "type": "nominal", "title": "Config"},
                {"field": "time", "type": "quantitative", "format": ".3f", "title": "Time (s)"},
                {"field": "cost", "type": "quantitative", "format": ".4f", "title": "Cost (USD)"},
                {"field": "speedup", "type": "quantitative", "format": ".3f", "title": "Speedup"},
                {"field": "cost_delta", "type": "quantitative", "format": ".3f", "title": "Cost delta vs control"},
                {"field": "status", "type": "nominal", "title": "Status"},
            ],
        },
    }

    layers: list[dict[str, Any]] = [scatter_layer]

    if frontier:
        layers.append(
            {
                "data": {"values": frontier},
                "mark": {"type": "line", "strokeDash": [6, 4], "strokeWidth": 2, "color": "#ff7f0e"},
                "encoding": {
                    "x": {"field": "time", "type": "quantitative"},
                    "y": {"field": "cost", "type": "quantitative"},
                },
            }
        )

    if cap_cost > 0:
        layers.append(
            {
                "data": {"values": [{}]},
                "mark": {"type": "rule", "strokeDash": [4, 4], "color": "#d62728"},
                "encoding": {
                    "y": {"datum": cap_cost, "type": "quantitative"},
                },
            }
        )

    spec: dict[str, Any] = {"layer": layers}
    st.vega_lite_chart(spec, use_container_width=True)


def _format_money(value: float) -> str:
    return f"${value:,.4f}"


def _render_result(result: DashboardResult, cost_cap_ratio: float) -> None:
    best, alternatives = _pick_recommendations(
        _to_points(result.control_run, result.validation_results, set(), result.run_summary.best_overall_config_id, cost_cap_ratio),
        max_alternatives=2,
    )

    selected_ids = {alt["id"] for alt in alternatives}
    if best is not None:
        selected_ids.add(best["id"])

    points = _to_points(
        result.control_run,
        result.validation_results,
        selected_ids,
        result.run_summary.best_overall_config_id,
        cost_cap_ratio,
    )
    frontier = _pareto_frontier(points)

    baseline_time = result.control_run.wall_clock_seconds
    baseline_cost = result.control_run.cost_estimate_usd
    cap_cost = baseline_cost * (1.0 + cost_cap_ratio)

    best_time = best["time"] if best else 0.0
    best_cost = best["cost"] if best else 0.0
    speed_gain = ((baseline_time - best_time) / baseline_time * 100.0) if best and baseline_time > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Baseline Time", f"{baseline_time:,.3f}s")
    m2.metric("Baseline Cost", _format_money(baseline_cost))
    m3.metric("Best Time-First Choice", f"{best_time:,.3f}s", f"{speed_gain:,.2f}%")
    m4.metric("Best Choice Cost", _format_money(best_cost), f"Cap {_format_money(cap_cost)}")

    st.subheader("Cost x Time Decision Surface")
    _render_chart(points, frontier, cap_cost)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("All Candidates")
        rows = [
            {
                "id": p["id"],
                "name": p["name"],
                "time_s": round(p["time"], 4),
                "cost_usd": round(p["cost"], 5),
                "status": p["status"],
                "eligible": p["eligible"],
                "success": p["success"],
                "diverged": p["diverged"],
                "speedup": round(p["speedup"], 4),
                "cost_delta_vs_control": round(p["cost_delta"], 4),
            }
            for p in points
        ]
        st.dataframe(rows, use_container_width=True)

    with right:
        st.subheader("Recommended")
        if best is None:
            st.warning("No valid non-diverged candidate meets the active cost cap.")
        else:
            st.success(f"Primary: {best['name']}")
            st.caption(
                f"Time: {best['time']:.3f}s | Cost: {_format_money(best['cost'])} | Speedup: {best['speedup']:.3f}x"
            )
            if alternatives:
                for idx, alt in enumerate(alternatives, start=1):
                    st.info(
                        f"Alt {idx}: {alt['name']} | {alt['time']:.3f}s | {_format_money(alt['cost'])} | {alt['speedup']:.3f}x"
                    )

        diverged = [p for p in points if p["diverged"] == "yes"]
        if diverged:
            st.warning(f"{len(diverged)} diverged configuration(s) detected and flagged in chart/table.")

    st.subheader("Run Summary")
    summary = {
        "best_overall_config_id": result.run_summary.best_overall_config_id,
        "best_native_baseline_id": result.run_summary.best_native_baseline_id,
        "speedup_vs_best_native": result.run_summary.speedup_vs_best_native,
        "cost_delta_vs_best_native": result.run_summary.cost_delta_vs_best_native,
        "throughput_gain_vs_best_native": result.run_summary.throughput_gain_vs_best_native,
        "objective_profile": result.run_summary.objective_profile,
        "confidence": result.run_summary.confidence,
        "recommended_gpu": result.run_summary.recommended_gpu,
    }
    st.json(summary)

    with st.expander("Evidence Edges"):
        if result.run_summary.evidence_edges:
            edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "detail": e.detail,
                }
                for e in result.run_summary.evidence_edges
            ]
            st.dataframe(edges, use_container_width=True)
        else:
            st.caption("No evidence edges recorded.")

    with st.expander("Artifacts"):
        st.write(f"Report: {result.report_path}")
        st.write(f"Configs: {result.config_export_dir}")

    # Deploy section
    st.subheader("Deploy Winning Config")
    best_config_id = result.run_summary.best_overall_config_id
    if best_config_id:
        config_json_path = result.config_export_dir / f"{best_config_id}.json"
        if config_json_path.exists():
            deploy_col1, deploy_col2 = st.columns([2, 1])
            with deploy_col1:
                deploy_steps = st.number_input(
                    "Training steps",
                    min_value=100,
                    max_value=1_000_000,
                    value=1000,
                    step=100,
                    help="Number of training steps for the full training run.",
                )
                deploy_gpu = st.selectbox(
                    "Deploy GPU",
                    ["a100-80gb", "a100-40gb", "h100", "a10g"],
                    key="deploy_gpu",
                )
            with deploy_col2:
                st.caption(f"Config: {best_config_id}")
                st.caption(f"File: {config_json_path}")
                deploy_clicked = st.button(
                    "Deploy on Modal",
                    type="primary",
                    use_container_width=True,
                )

            if deploy_clicked:
                deploy_status = st.empty()
                deploy_log = st.empty()
                deploy_status.info("Deploying winning config on Modal...")
                try:
                    with open(config_json_path) as f:
                        opt_config = OptimizationConfig.from_dict(json.load(f))

                    from gpunity.validator.patcher import apply_config

                    # Get repo_path from session - it's in the sidebar config
                    repo_path = Path(st.session_state.get("_last_repo_path", "."))
                    patched_repo = apply_config(repo_path, opt_config)

                    entry_point = st.session_state.get("_last_entry_point", "train.py")
                    train_steps_env = f'os.environ["TRAIN_STEPS"] = "{deploy_steps}"'
                    wrapper_script = f'''#!/usr/bin/env python3
"""GPUnity deploy wrapper."""
import importlib.util, os, sys, time
repo_dir = os.environ.get("REPO_DIR", os.getcwd())
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)
{train_steps_env}
print("[gpunity deploy] Starting optimized training...")
start = time.time()
spec = importlib.util.spec_from_file_location("__train_module__", os.path.join(repo_dir, "{entry_point}"))
module = importlib.util.module_from_spec(spec)
sys.modules["__train_module__"] = module
spec.loader.exec_module(module)
for candidate in ("train", "main", "run"):
    fn = getattr(module, candidate, None)
    if callable(fn):
        fn()
        break
elapsed = time.time() - start
print(f"[gpunity deploy] Training complete in {{elapsed:.1f}}s")
'''
                    from gpunity.sandbox.modal_runner import run_in_modal

                    artifact_dir = _run_coro(run_in_modal(
                        repo_path=patched_repo,
                        script_content=wrapper_script,
                        gpu_type=deploy_gpu,
                        timeout_seconds=3600,
                        stream_logs=True,
                    ))
                    deploy_status.success(f"Deploy complete! Artifacts: {artifact_dir}")
                except Exception as exc:
                    deploy_status.error(f"Deploy failed: {exc}")
        else:
            st.warning(f"Config file not found: {config_json_path}")
    else:
        st.info("No winning config to deploy. Run the optimization pipeline first.")


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Run Controls")

    _default_repo = str(Path(__file__).resolve().parents[2] / "sample_codebases" / "nano_gpt")
    repo_path = st.sidebar.text_input("Repo path", value=_default_repo)
    entry_point = st.sidebar.text_input("Entry point", value="gpunity_entry_nanogpt.py")
    output_path = st.sidebar.text_input("Report path", value="./logs/streamlit_report.md")
    provider = st.sidebar.selectbox(
        "Provider", ["openai", "heuristic", "modal", "claude"], index=0
    )

    model = None
    modal_vllm_url = ""
    if provider == "openai":
        model = st.sidebar.selectbox(
            "Model", ["gpt-5.2", "gpt-5.2-mini", "gpt-4o"], index=0
        )
    elif provider == "claude":
        model = st.sidebar.selectbox(
            "Model",
            ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
            index=0,
        )
    elif provider == "modal":
        modal_vllm_url = st.sidebar.text_input(
            "Modal vLLM URL",
            value=os.environ.get("MODAL_VLLM_URL", ""),
            help="Deploy modal_app.py and paste the endpoint URL here.",
        )
        st.sidebar.text_input("Model", value="Qwen/Qwen3-8B", disabled=True)
    objective_profile = st.sidebar.selectbox(
        "Objective profile",
        ["latency", "balanced", "cost", "throughput"],
        index=0,
    )
    validation_strategy = st.sidebar.selectbox(
        "Validation strategy",
        ["parallel_all", "staged"],
        index=0,
    )
    gpu_type = st.sidebar.selectbox("GPU type", ["a100-80gb", "a100-40gb", "h100", "a10g"], index=0)
    local_mode = st.sidebar.checkbox("Local sandbox", value=False)
    verbose = st.sidebar.checkbox("Verbose logs", value=False)
    profile_steps = st.sidebar.slider("Profile steps", min_value=5, max_value=200, value=20)
    validation_steps = st.sidebar.slider("Validation steps", min_value=10, max_value=400, value=50)
    top_k = st.sidebar.slider("Top-k validate", min_value=1, max_value=20, value=5)
    max_configs = st.sidebar.slider("Max generated configs", min_value=1, max_value=30, value=10)
    cost_cap_pct = st.sidebar.slider(
        "Cost cap over baseline (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
    )

    run_clicked = st.sidebar.button("Run Optimization", type="primary", use_container_width=True)

    return {
        "run_clicked": run_clicked,
        "repo_path": repo_path,
        "entry_point": entry_point,
        "output_path": output_path,
        "provider": provider,
        "model": model,
        "modal_vllm_url": modal_vllm_url,
        "objective_profile": objective_profile,
        "validation_strategy": validation_strategy,
        "gpu_type": gpu_type,
        "local": local_mode,
        "verbose": verbose,
        "profile_steps": profile_steps,
        "validation_steps": validation_steps,
        "top_k": top_k,
        "max_configs": max_configs,
        "cost_cap_ratio": cost_cap_pct / 100.0,
    }


def _build_demo_result() -> DashboardResult:
    """Provide a synthetic demo dataset so the dashboard renders without a real run."""
    control = ControlRun(
        steps_completed=50,
        wall_clock_seconds=12.4,
        avg_step_time_ms=248.0,
        peak_memory_mb=4200.0,
        throughput_samples_sec=32.3,
        loss_values=[2.8, 2.5, 2.3, 2.1, 2.0],
        gradient_norms=[1.2, 1.1, 1.0, 0.95, 0.9],
        cost_estimate_usd=0.0082,
    )

    from gpunity.types import EvidenceEdge

    results = [
        ValidationResult(
            config_id="opt-001",
            config_name="torch.compile (default)",
            success=True,
            speedup_vs_control=1.45,
            throughput_gain_vs_control=0.42,
            memory_delta_vs_control=-0.05,
            cost_delta_vs_control=0.08,
            wall_clock_seconds=8.55,
        ),
        ValidationResult(
            config_id="opt-002",
            config_name="TF32 matmul",
            success=True,
            speedup_vs_control=1.22,
            throughput_gain_vs_control=0.20,
            memory_delta_vs_control=0.0,
            cost_delta_vs_control=-0.02,
            wall_clock_seconds=10.16,
        ),
        ValidationResult(
            config_id="opt-003",
            config_name="FlashAttention-2",
            success=True,
            speedup_vs_control=1.65,
            throughput_gain_vs_control=0.58,
            memory_delta_vs_control=-0.30,
            cost_delta_vs_control=0.15,
            wall_clock_seconds=7.52,
        ),
        ValidationResult(
            config_id="opt-004",
            config_name="Aggressive compile + AMP",
            success=True,
            diverged=True,
            divergence_step=22,
            divergence_reason="Loss NaN after step 22",
            speedup_vs_control=1.80,
            throughput_gain_vs_control=0.72,
            memory_delta_vs_control=-0.10,
            cost_delta_vs_control=0.25,
            wall_clock_seconds=6.89,
        ),
        ValidationResult(
            config_id="opt-005",
            config_name="Channels-last memory format",
            success=True,
            speedup_vs_control=1.10,
            throughput_gain_vs_control=0.09,
            memory_delta_vs_control=-0.02,
            cost_delta_vs_control=-0.01,
            wall_clock_seconds=11.27,
        ),
    ]

    summary = RunSummary(
        objective_profile="latency",
        best_overall_config_id="opt-003",
        best_native_baseline_id="B0",
        speedup_vs_best_native=1.65,
        cost_delta_vs_best_native=0.15,
        throughput_gain_vs_best_native=0.58,
        confidence="medium",
        evidence_edges=[
            EvidenceEdge(
                source="top_operators[0]: aten::scaled_dot_product_attention",
                target="opt-003",
                relation="triggers",
                detail="60% GPU time in attention -> FlashAttention-2",
            ),
            EvidenceEdge(
                source="gpu_utilization: 45%",
                target="opt-001",
                relation="triggers",
                detail="Low GPU util -> torch.compile fuses ops",
            ),
        ],
    )

    return DashboardResult(
        report_path=Path("./logs/demo_report.md"),
        config_export_dir=Path("./logs/gpunity_configs"),
        control_run=control,
        validation_results=results,
        run_summary=summary,
    )


def main() -> None:
    st.set_page_config(page_title="GPUnity Command Center", layout="wide")
    st.title("GPUnity Optimization")
    st.caption("Time-first optimization under explicit cost guardrails for fast demo decisions.")

    sidebar = _render_sidebar()

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    if sidebar["run_clicked"]:
        # Set Modal vLLM URL in env if provided
        if sidebar.get("modal_vllm_url"):
            os.environ["MODAL_VLLM_URL"] = sidebar["modal_vllm_url"]

        cli_args = {
            "repo_path": sidebar["repo_path"],
            "output_path": sidebar["output_path"],
            "entry_point": sidebar["entry_point"],
            "provider": sidebar["provider"],
            "model": sidebar["model"],
            "objective_profile": sidebar["objective_profile"],
            "validation_strategy": sidebar["validation_strategy"],
            "gpu_type": sidebar["gpu_type"],
            "local": sidebar["local"],
            "verbose": sidebar["verbose"],
            "profile_steps": sidebar["profile_steps"],
            "validation_steps": sidebar["validation_steps"],
            "top_k": sidebar["top_k"],
            "max_configs": sidebar["max_configs"],
            "compare_against_native": True,
            "baseline_policy": "required",
            "max_cost_per_sandbox": 5.0,
            "dry_run": False,
            "gpu_selection": "auto",
            "gpu_aggressiveness": "moderate",
        }

        try:
            config = load_config(cli_args)
            status_placeholder = st.empty()
            status_placeholder.info("Running GPUnity pipeline...")
            result = _run_coro(_execute_pipeline_phased(config, status_placeholder))
            status_placeholder.success("Pipeline complete!")
            st.session_state.last_result = result
            st.session_state._last_repo_path = sidebar["repo_path"]
            st.session_state._last_entry_point = sidebar["entry_point"]
            st.session_state.last_error = None
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.session_state.last_result = None

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    result = st.session_state.last_result

    if result is None:
        st.info("Configure settings in the sidebar and press **Run Optimization** to start.")
    else:
        _render_result(result, sidebar["cost_cap_ratio"])


if __name__ == "__main__":
    main()
