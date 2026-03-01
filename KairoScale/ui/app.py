"""Streamlit command center for KairoScale optimization runs."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import zipfile
from datetime import datetime
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

# Ensure the project root is importable even when invoked via
# `streamlit run KairoScale/ui/app.py` without `pip install -e .`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from KairoScale.config import load_config
from KairoScale.types import (
    ControlRun,
    OptimizationConfig,
    RunSummary,
    ValidationResult,
    are_configs_compatible,
    merge_configs,
)


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


def _load_configs_from_dir(config_dir: Path) -> dict[str, OptimizationConfig]:
    loaded: dict[str, OptimizationConfig] = {}
    if not config_dir.exists():
        return loaded
    for config_path in sorted(config_dir.glob("*.json")):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = OptimizationConfig.from_dict(json.load(f))
            loaded[config.id] = config
        except Exception:
            continue
    return loaded


def _append_run_log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    logs = st.session_state.setdefault("run_logs", [])
    logs.append(f"[{timestamp}] {message}")


def _render_logs_panel(enabled: bool) -> None:
    if not enabled:
        return
    st.subheader("Run Logs")
    logs = st.session_state.get("run_logs", [])
    if not logs:
        st.caption("No logs yet. Start a run to stream phase updates here.")
        return
    st.code("\n".join(logs), language="text")


def _build_configs_zip(config_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        if config_dir.exists():
            for cfg_path in sorted(config_dir.glob("*.json")):
                archive.writestr(cfg_path.name, cfg_path.read_text(encoding="utf-8"))
    return buffer.getvalue()


def _render_phase_guide() -> None:
    with st.expander("How The Pipeline Works", expanded=True):
        st.markdown(
            """
**Phase 1 - Profile**
- Runs your training entry point in a sandbox and captures runtime metrics.
- Outputs baseline stats like utilization, memory, and step timing.

**Phase 2 - Analyze**
- Uses the selected provider to diagnose bottlenecks from profile evidence.
- Proposes optimization configs with rationale and risk level.

**Phase 3 - Validate**
- Replays control + candidate configs in parallel sandboxes.
- Measures speed, cost, memory, and correctness signals before ranking.

**Phase 4 - Report**
- Produces an executive summary, comparison tables, and recommended next steps.
"""
        )


async def _execute_pipeline_phased(
    config: Any,
    status_container,
    log_callback: Any | None = None,
) -> DashboardResult:
    from KairoScale.agent.combo import generate_combo_configs
    from KairoScale.agent.loop import run_agent_loop
    from KairoScale.cli import run_profile_phase
    from KairoScale.diagnosis.bottleneck import diagnose_bottleneck
    from KairoScale.hardware.profile import detect_hardware_profile, resolve_workload_mode
    from KairoScale.reporter.markdown import generate_report
    from KairoScale.validator.runner import _run_validation_batch, run_validation

    sandbox_mode = "Modal Sandbox" if not config.local else "Local Sandbox"
    provider_label = config.provider.capitalize()

    def log_step(message: str) -> None:
        status_container.info(message)
        if log_callback is not None:
            log_callback(message)

    log_step(f"Phase 1: Profiling training script in {sandbox_mode}...")
    profile_result = await run_profile_phase(config)
    gpu_util = getattr(profile_result, "gpu_utilization", None)
    peak_mem = getattr(profile_result, "peak_memory_mb", None)
    phase1_msg = "Phase 1 complete."
    if gpu_util is not None:
        phase1_msg += f" GPU util: {gpu_util:.0f}%"
    if peak_mem is not None:
        phase1_msg += f", Peak VRAM: {peak_mem:.0f} MB"
    log_step(phase1_msg)

    hardware = detect_hardware_profile(config)
    mode = resolve_workload_mode(config.mode, profile_result)

    diagnosis = None
    if config.gpu_selection == "auto":
        diagnosis = diagnose_bottleneck(profile_result, hardware, config.gpu_aggressiveness)
        if diagnosis:
            log_step(
                f"Bottleneck: {diagnosis.primary.value} (confidence: {diagnosis.confidence})"
            )

    if config.provider == "modal" and not os.environ.get("MODAL_VLLM_URL", "").strip():
        log_step("Auto-deploying vLLM on Modal (2-5 min)...")

    log_step(f"Phase 2: {provider_label} LLM analyzing profile and proposing optimizations...")
    optimization_configs = await run_agent_loop(
        profile_result,
        Path(config.repo_path),
        config,
        hardware_profile=hardware,
        mode=mode,
        diagnosis=diagnosis,
    )
    log_step(f"Phase 2 complete. Generated {len(optimization_configs)} optimization configs.")

    config_export_dir = _export_configs(
        optimization_configs,
        Path(config.output_path).parent / "KairoScale_configs",
    )

    log_step(
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
    log_step(
        f"Phase 3 complete. {len(validation_results)} validated, {n_diverged} diverged."
    )

    log_step("Phase 3.5: Generating and validating combo configs...")
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
        log_step(
            f"Phase 3.5 complete. Validated {len(combo_results)} combo configs."
        )
    else:
        log_step("Phase 3.5 complete. No compatible combos generated.")

    log_step("Phase 4: Generating report...")
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


def _validate_combo_config(combo: OptimizationConfig, control: ControlRun) -> ValidationResult:
    from KairoScale.validator.divergence import compute_cosine_similarities
    from KairoScale.validator.gradient_tracker import create_gradient_tracking_wrapper
    from KairoScale.validator.metrics import compute_validation_metrics
    from KairoScale.validator.patcher import apply_config

    run_cfg = st.session_state.get("_last_run_config", {})
    repo_path = Path(run_cfg.get("repo_path", st.session_state.get("_last_repo_path", ".")))
    entry_point = str(run_cfg.get("entry_point", st.session_state.get("_last_entry_point", "train.py")))
    mode = str(run_cfg.get("mode", "train"))
    steps = int(run_cfg.get("validation_steps", 50))
    gradient_check_interval = int(run_cfg.get("gradient_check_interval", 5))
    validation_seed = int(run_cfg.get("validation_seed", 1337))
    deterministic_validation = bool(run_cfg.get("deterministic_validation", True))
    gpu_type = str(run_cfg.get("gpu_type", "a100-80gb"))
    max_cost_per_sandbox = float(run_cfg.get("max_cost_per_sandbox", 5.0))
    timeout_seconds = max(600, min(5400, 180 + steps * 20))

    patched_repo = apply_config(repo_path, combo)
    wrapper_script = create_gradient_tracking_wrapper(
        entry_point=entry_point,
        steps=steps,
        gradient_check_interval=gradient_check_interval,
        artifact_dir=Path("/tmp/KairoScale_val"),
        mode=mode,
        validation_seed=validation_seed,
        deterministic_validation=deterministic_validation,
    )

    if bool(run_cfg.get("local", False)):
        from KairoScale.sandbox.local_runner import run_locally

        artifact_dir = _run_coro(
            run_locally(
                repo_path=patched_repo,
                script_content=wrapper_script,
                timeout_seconds=timeout_seconds,
                python_bin=run_cfg.get("python_bin"),
            )
        )
    else:
        from KairoScale.sandbox.modal_runner import run_in_modal

        artifact_dir = _run_coro(
            run_in_modal(
                repo_path=patched_repo,
                script_content=wrapper_script,
                gpu_type=gpu_type,
                timeout_seconds=timeout_seconds,
                cost_ceiling_usd=max_cost_per_sandbox,
            )
        )

    result = compute_validation_metrics(
        artifact_dir,
        control,
        config_id=combo.id,
        config_name=combo.name,
        gpu_type=gpu_type,
    )
    result.gradient_cosine_similarities = compute_cosine_similarities(
        control.loss_values,
        result.loss_values,
    )
    return result


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

        if result.report_path.exists():
            st.download_button(
                "Download report (.md)",
                data=result.report_path.read_bytes(),
                file_name=result.report_path.name,
                mime="text/markdown",
                key=f"download_report_{result.report_path.name}",
                use_container_width=True,
            )
        else:
            st.caption("Report file is not available yet.")

        if result.config_export_dir.exists():
            st.download_button(
                "Download all configs (.zip)",
                data=_build_configs_zip(result.config_export_dir),
                file_name="KairoScale_configs.zip",
                mime="application/zip",
                key="download_configs_zip",
                use_container_width=True,
            )

            winner_id = result.run_summary.best_overall_config_id
            if winner_id:
                winner_path = result.config_export_dir / f"{winner_id}.json"
                if winner_path.exists():
                    st.download_button(
                        "Download winning config (.json)",
                        data=winner_path.read_bytes(),
                        file_name=winner_path.name,
                        mime="application/json",
                        key=f"download_winner_{winner_id}",
                        use_container_width=True,
                    )
                    apply_repo = st.session_state.get("_last_repo_path", ".")
                    st.caption("Apply winner in your repo")
                    st.code(
                        f"KairoScale apply {winner_path} --repo {apply_repo}",
                        language="bash",
                    )

                    if st.button(
                        "Apply Winning Config to Repo",
                        key=f"apply_winner_in_place_{winner_id}",
                        use_container_width=True,
                    ):
                        apply_status = st.empty()
                        try:
                            from KairoScale.validator.patcher import apply_config_in_place

                            with open(winner_path, encoding="utf-8") as f:
                                winning_config = OptimizationConfig.from_dict(json.load(f))

                            written_files = apply_config_in_place(Path(apply_repo), winning_config)
                            if written_files:
                                apply_status.success(
                                    f"Applied {winning_config.id} to {apply_repo}. "
                                    f"Wrote {len(written_files)} file(s)."
                                )
                                st.code(
                                    "\n".join(str(p) for p in written_files),
                                    language="text",
                                )
                            else:
                                apply_status.info(
                                    f"Applied {winning_config.id} to {apply_repo}. "
                                    "No files were changed."
                                )
                        except Exception as exc:
                            apply_status.error(f"Apply failed: {exc}")

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

                    from KairoScale.validator.patcher import apply_config

                    # Get repo_path from session - it's in the sidebar config
                    repo_path = Path(st.session_state.get("_last_repo_path", "."))
                    patched_repo = apply_config(repo_path, opt_config)

                    entry_point = st.session_state.get("_last_entry_point", "train.py")
                    train_steps_env = f'os.environ["TRAIN_STEPS"] = "{deploy_steps}"'
                    wrapper_script = f'''#!/usr/bin/env python3
"""KairoScale deploy wrapper."""
import importlib.util, os, sys, time
repo_dir = os.environ.get("REPO_DIR", os.getcwd())
sys.path.insert(0, repo_dir)
os.chdir(repo_dir)
{train_steps_env}
print("[KairoScale deploy] Starting optimized training...")
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
print(f"[KairoScale deploy] Training complete in {{elapsed:.1f}}s")
'''
                    from KairoScale.sandbox.modal_runner import run_in_modal

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

    st.subheader("Build Custom Combo")
    loaded_configs = _load_configs_from_dir(result.config_export_dir)
    valid_results = [
        r
        for r in result.validation_results
        if r.success and not r.diverged and r.config_id in loaded_configs
    ]
    if len(valid_results) < 2:
        st.caption("Need at least two validated non-diverged configs to build a combo.")
        return

    option_labels = [
        f"{r.config_name} ({r.config_id})"
        for r in valid_results
    ]
    label_to_id = {
        f"{r.config_name} ({r.config_id})": r.config_id
        for r in valid_results
    }
    selected_combo_labels = st.multiselect(
        "Select compatible optimizations",
        options=option_labels,
        help="Choose at least two validated configs to combine.",
    )
    selected_combo_ids = [label_to_id[label] for label in selected_combo_labels]

    if len(selected_combo_ids) < 2:
        st.caption("Select at least two configs to run compatibility checks.")
        return

    selected_configs = [loaded_configs[cid] for cid in selected_combo_ids]
    pair_results: list[tuple[OptimizationConfig, OptimizationConfig, bool, str]] = []
    all_compatible = True
    for left_cfg, right_cfg in combinations(selected_configs, 2):
        compatible, reason = are_configs_compatible(left_cfg, right_cfg)
        pair_results.append((left_cfg, right_cfg, compatible, reason))
        if not compatible:
            all_compatible = False

    for left_cfg, right_cfg, compatible, reason in pair_results:
        line = f"{left_cfg.name} + {right_cfg.name}"
        if compatible:
            st.success(f"{line}: compatible")
        else:
            st.error(f"{line}: incompatible ({reason})")

    if not all_compatible:
        st.warning("Resolve incompatible pairs before validating a custom combo.")
        return

    try:
        merged_combo = merge_configs(selected_configs)
    except ValueError as exc:
        st.error(f"Could not merge selected configs: {exc}")
        return

    st.caption("Merged combo preview")
    st.json(
        {
            "id": merged_combo.id,
            "name": merged_combo.name,
            "optimization_type": merged_combo.optimization_type.value,
            "estimated_speedup": merged_combo.estimated_speedup,
            "estimated_memory_delta": merged_combo.estimated_memory_delta,
            "risk_level": merged_combo.risk_level.value,
            "config_overrides": merged_combo.config_overrides,
        }
    )

    if st.button("Validate Combo", type="primary", key="validate_custom_combo"):
        status = st.empty()
        status.info("Validating custom combo...")
        try:
            combo_result = _validate_combo_config(merged_combo, result.control_run)
            result.config_export_dir.mkdir(parents=True, exist_ok=True)
            combo_path = result.config_export_dir / f"{merged_combo.id}.json"
            combo_path.write_text(json.dumps(merged_combo.to_dict(), indent=2), encoding="utf-8")
            result.validation_results = [
                r for r in result.validation_results if r.config_id != combo_result.config_id
            ]
            result.validation_results.append(combo_result)
            st.session_state.last_result = result
            if combo_result.success and not combo_result.diverged:
                status.success(
                    "Combo validated successfully: "
                    f"{combo_result.speedup_vs_control:.2f}x speedup vs control."
                )
            elif combo_result.success:
                status.error(
                    "Combo run diverged: "
                    f"{combo_result.divergence_reason or 'failed stability checks'}"
                )
            else:
                status.error(f"Combo validation failed: {combo_result.error}")
        except Exception as exc:
            status.error(f"Combo validation failed: {exc}")


def _render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Run Controls")

    _default_repo = str(Path(__file__).resolve().parents[2] / "sample_codebases" / "nano_gpt")
    repo_path = st.sidebar.text_input("Repo path", value=_default_repo)
    entry_point = st.sidebar.text_input("Entry point", value="KairoScale_entry_nanogpt.py")
    output_path = st.sidebar.text_input("Report path", value="./logs/streamlit_report.md")
    provider = st.sidebar.selectbox(
        "Provider", ["modal", "openai", "claude", "heuristic"], index=0
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
        default_modal_url = st.session_state.get(
            "modal_vllm_url",
            os.environ.get("MODAL_VLLM_URL", ""),
        )
        modal_vllm_url = st.sidebar.text_input(
            "Modal vLLM URL",
            value=default_modal_url,
            help="Leave blank to auto-deploy.",
        )
        st.sidebar.text_input("Model", value="Qwen/Qwen3-8B", disabled=True)
        if not modal_vllm_url.strip():
            st.sidebar.info("vLLM will be auto-deployed on Modal when you run.")
    objective_profile = st.sidebar.selectbox(
        "Objective profile",
        ["balanced", "latency", "cost", "throughput"],
        index=0,
    )
    validation_strategy = st.sidebar.selectbox(
        "Validation strategy",
        ["parallel_all", "staged"],
        index=0,
    )
    gpu_type = st.sidebar.selectbox("GPU type", ["a100-80gb", "a100-40gb", "h100", "a10g"], index=0)
    local_mode = st.sidebar.checkbox("Local sandbox", value=False)
    profile_steps = st.sidebar.slider("Profile steps", min_value=5, max_value=200, value=80)
    validation_steps = st.sidebar.slider("Validation steps", min_value=10, max_value=400, value=150)
    top_k = st.sidebar.slider("Top-k validate", min_value=1, max_value=20, value=8)
    max_configs = st.sidebar.slider("Max generated configs", min_value=1, max_value=30, value=24)
    cost_cap_pct = st.sidebar.slider(
        "Cost cap over baseline (%)",
        min_value=0,
        max_value=100,
        value=100,
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

    from KairoScale.types import EvidenceEdge

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
        config_export_dir=Path("./logs/KairoScale_configs"),
        control_run=control,
        validation_results=results,
        run_summary=summary,
    )


def main() -> None:
    st.set_page_config(page_title="KairoScale Command Center", layout="wide")
    st.title("KairoScale Optimization")
    st.caption("Optimization with configurable objectives and explicit cost guardrails")
    _render_phase_guide()

    sidebar = _render_sidebar()

    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = True

    if sidebar["run_clicked"]:
        st.session_state.run_logs = []
        _append_run_log("Run requested.")
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
            "verbose": False,
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
            st.session_state._last_run_config = config.to_dict()
            status_placeholder = st.empty()
            status_placeholder.info("Running KairoScale pipeline...")
            _append_run_log("Running KairoScale pipeline...")
            result = _run_coro(
                _execute_pipeline_phased(
                    config,
                    status_placeholder,
                    log_callback=_append_run_log,
                )
            )
            status_placeholder.success("Pipeline complete!")
            _append_run_log("Pipeline complete.")
            st.session_state.last_result = result
            st.session_state._last_repo_path = sidebar["repo_path"]
            st.session_state._last_entry_point = sidebar["entry_point"]
            if sidebar["provider"] == "modal":
                deployed_url = os.environ.get("MODAL_VLLM_URL", "").strip()
                if deployed_url:
                    st.session_state["modal_vllm_url"] = deployed_url
            st.session_state.last_error = None
        except Exception as exc:
            st.session_state.last_error = str(exc)
            st.session_state.last_result = None
            _append_run_log(f"Pipeline failed: {exc}")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    log_col1, log_col2 = st.columns(2)
    with log_col1:
        if st.button("Show Logs", key="show_logs_button", use_container_width=True):
            st.session_state.show_logs = True
    with log_col2:
        if st.button("Hide Logs", key="hide_logs_button", use_container_width=True):
            st.session_state.show_logs = False

    _render_logs_panel(st.session_state.show_logs)

    result = st.session_state.last_result

    if result is None:
        st.info("Configure settings in the sidebar and press **Run Optimization** to start.")
    else:
        _render_result(result, sidebar["cost_cap_ratio"])


if __name__ == "__main__":
    main()
