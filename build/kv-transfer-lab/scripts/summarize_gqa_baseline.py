# SPDX-License-Identifier: Apache-2.0
"""Audit saved baseline samples and export untrimmed statistics and figures.

Run in a separate plotting environment after GPU experiments have finished.
"""

# Standard
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import math
import statistics

# Third Party
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def stats(values: list[float]) -> dict[str, float | int | None]:
    """Summarize all finite observed values without removing outliers.

    Args:
        values: Observed latencies in milliseconds, in sample order.

    Returns:
        Median, nearest-rank tails, maximum, drift and spike count, or None
        for unavailable statistics. Empty input has zero observed values.
    """
    if not values:
        return dict.fromkeys(("median", "p95", "p99", "maximum", "drift", "spikes"))
    ordered = sorted(values)
    median = statistics.median(values)
    quarter = max(1, len(values) // 4)
    return {
        "median": median,
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "p99": ordered[math.ceil(0.99 * len(values)) - 1],
        "maximum": ordered[-1],
        "drift": abs(
            statistics.median(values[-quarter:]) - statistics.median(values[:quarter])
        )
        / median,
        "spikes": sum(value > 2 * median for value in values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows with a union of columns and explicit empty missing values.

    Args:
        path: New CSV destination.
        rows: Records whose numeric values are never silently filtered.

    Returns:
        None.
    """
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_run(
    run: dict[str, Any], data: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    """Check saved completion, validation, identity and geometry evidence.

    Args:
        run: Controller record, possibly an unattempted placeholder.
        data: Raw benchmark JSON, possibly empty if unavailable.
        plan: Frozen premeasurement plan.

    Returns:
        All detected inconsistencies; an empty list means validated evidence.
    """
    errors = []
    case = run["scenario"]
    arguments = data.get("arguments", {})
    expected = {
        **plan["geometry"],
        **{key: case[key] for key in ("direction", "mapping", "num_objects")},
        **{key: plan[key] for key in ("warmup", "iterations", "seed")},
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            errors.append(f"argument mismatch: {key}")
    geometry = plan["geometry"]
    width = {"bfloat16": 2, "float16": 2, "float32": 4}[geometry["dtype"]]
    effective_bytes = (
        2
        * geometry["layers"]
        * geometry["kv_heads"]
        * geometry["head_dim"]
        * width
        * geometry["chunk_size"]
        * case["num_objects"]
    )
    if run.get("exit_code") != 0 or data.get("passed") is not True:
        errors.append("process failed or unattempted")
    provenance = data.get("provenance", {})
    if provenance.get("effective_bytes_per_call") != effective_bytes:
        errors.append("effective payload bytes disagree with geometry")
    if provenance.get("git_sha") != plan["git_sha"]:
        errors.append("Git SHA mismatch")
    if provenance.get("git_dirty_status") != "":
        errors.append("dirty or unavailable checkout")
    hashes = provenance.get("files_sha256", {})
    benchmark_hashes = [v for k, v in hashes.items() if k.endswith(plan["benchmark"])]
    if benchmark_hashes != [plan["script_sha256"]]:
        errors.append("benchmark hash mismatch")
    phases = data.get("validations", [])
    if [row.get("phase") for row in phases] != [
        "precheck",
        "after_warmup",
        "after_measurement",
    ]:
        errors.append("validation phases missing or reordered")
    checks = (
        "all_gpu_layers_exact",
        "host_objects_exact",
        "unselected_gpu_blocks_unchanged",
        "host_guards_unchanged",
        "source_unchanged",
    )
    if any(row.get(key) is not True for row in phases for key in checks):
        errors.append("data, sentinel, guard or source validation failed")
    if phases and any(
        row.get("object_sha256") != phases[0].get("object_sha256") for row in phases
    ):
        errors.append("object checksums changed")
    calls = 1 + plan["warmup"] + plan["iterations"]
    batches = math.ceil(case["num_objects"] / 4)
    for key, value in {
        "native_calls_executed": calls * batches,
        "destination_resets_executed": calls,
        "warmup_calls_executed": plan["warmup"],
        "measured_calls_executed": plan["iterations"],
    }.items():
        if data.get(key) != value:
            errors.append(f"counter mismatch: {key}")
    buffers = data.get("buffers", {})
    pool = provenance.get("pool_blocks", 0)
    block_size, heads, dim = (
        geometry["block_size"],
        geometry["kv_heads"],
        geometry["head_dim"],
    )
    gpu_shape = [2, pool, block_size, heads, dim]
    gpu_stride = [
        pool * block_size * heads * dim,
        block_size * heads * dim,
        heads * dim,
        dim,
        1,
    ]
    layers, chunk = geometry["layers"], geometry["chunk_size"]
    host_shape = [2, layers, chunk, heads * dim]
    host_stride = [layers * chunk * heads * dim, chunk * heads * dim, heads * dim, 1]
    if (
        buffers.get("gpu_shapes") != [gpu_shape] * layers
        or buffers.get("gpu_strides") != [gpu_stride] * layers
        or buffers.get("host_shapes") != [host_shape] * case["num_objects"]
        or buffers.get("host_strides") != [host_stride] * case["num_objects"]
    ):
        errors.append("shape or stride disagrees with declared NHD layout")
    ids = buffers.get("block_ids", [])
    if (
        len(ids) != provenance.get("num_selected_blocks")
        or len(set(ids)) != len(ids)
        or any(not 0 <= value < provenance.get("pool_blocks", 0) for value in ids)
        or buffers.get("unique_scatter_targets") != len(ids)
        or buffers.get("all_host_objects_pinned") is not True
    ):
        errors.append("mapping uniqueness, bounds or pinning invalid")
    samples = data.get("samples", [])
    if len(samples) != plan["iterations"] or [
        sample.get("index") for sample in samples
    ] != list(range(plan["iterations"])):
        errors.append("missing, extra or reordered samples")
    if any(
        sample.get("completion_event_ready") is not True
        or sample.get("destination_reset_executed") is not True
        or sample.get("native_calls") != batches
        or sample.get("effective_bytes") != provenance.get("effective_bytes_per_call")
        or not all(
            isinstance(sample.get(key), (int, float))
            and math.isfinite(sample[key])
            and sample[key] > 0
            for key in ("cpu_wall_ns", "device_stream_span_ms")
        )
        for sample in samples
    ):
        errors.append("invalid sample or incomplete/no-op call")
    return errors


def export_figures(
    output: Path,
    cases: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    raw: dict[str, dict[str, Any]],
    telemetry: list[dict[str, Any]],
) -> None:
    """Export PNG and SVG figures without clipping or trimming latency samples.

    Args:
        output: New analysis directory.
        cases: Ordered planned scenarios.
        runs: All run summaries, including invalid or unattempted runs.
        raw: Original benchmark records by run ID.
        telemetry: All monitoring records, with missing values retained.

    Returns:
        None. Telemetry figures describe whole process envelopes, not kernels.
    """
    charts = output / "charts"
    charts.mkdir()
    colors = ["#0072B2", "#D55E00", "#009E73"]
    labels = [case["id"].replace("-", "\n", 1) for case in cases]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), layout="constrained")
    for axis, metric in zip(axes, ("wall", "stream"), strict=True):
        for index, case in enumerate(cases):
            for run in runs:
                if run["scenario"] != case["id"]:
                    continue
                values = run[metric]
                if values["median"] is None:
                    continue
                x = index + (run["repeat"] - 2) * 0.15
                color = colors[run["repeat"] - 1]
                axis.plot([x, x], [values["median"], values["p99"]], color=color)
                axis.scatter(x, values["median"], color=color, marker="o")
                axis.scatter(x, values["p95"], color=color, marker="_")
                axis.scatter(x, values["p99"], color=color, marker="x")
        axis.set_ylabel(f"{metric} latency (ms)")
        axis.set_xticks(range(len(cases)), labels)
        for repeat, color in enumerate(colors, start=1):
            axis.plot([], [], marker="o", color=color, label=f"repeat {repeat}")
        axis.legend(loc="upper left")
        axis.grid(alpha=0.25)
    axes[0].set_title(
        "Native batch + completion: median (o), p95 (_), p99 (x); repeats 1/2/3"
    )
    axes[1].set_title("Stream event span includes host enqueue gaps; not kernel time")
    for extension in ("png", "svg"):
        fig.savefig(charts / f"latency-summary.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout="constrained")
    for axis, case in zip(axes.flat, cases, strict=True):
        for run in runs:
            if run["scenario"] != case["id"]:
                continue
            samples = raw[run["run_id"]].get("samples", [])
            axis.plot(
                [sample.get("index") for sample in samples],
                [sample.get("cpu_wall_ns", float("nan")) / 1e6 for sample in samples],
                lw=0.7,
                alpha=0.8,
                color=colors[run["repeat"] - 1],
                label=f"r{run['repeat']} ({run['status']})",
            )
        axis.set_title(case["id"])
        axis.set_yscale("log")
        axis.set_xlabel("Measured sample index")
        axis.set_ylabel("Wall latency (ms, log scale)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle("All recorded wall samples, no outlier removal; invalid runs labelled")
    for extension in ("png", "svg"):
        fig.savefig(charts / f"wall-samples.{extension}", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), layout="constrained")
    fields = [
        [("temperature.gpu", "Temperature (C)")],
        [("clocks.current.sm", "SM"), ("clocks.current.memory", "Memory")],
        [("power.draw", "Power (W)")],
    ]
    start = telemetry[0]["received_monotonic_ns"] if telemetry else 0
    for axis, selections in zip(axes, fields, strict=True):
        for field, label in selections:
            times, values = [], []
            for row in telemetry:
                times.append((row["received_monotonic_ns"] - start) / 1e9)
                try:
                    value = float(row.get("values", {}).get(field, "nan"))
                except ValueError:
                    value = float("nan")
                values.append(value)
            axis.plot(times, values, label=label, lw=0.8)
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlabel("Monitor elapsed (s)")
    axes[0].set_ylabel("C")
    axes[1].set_ylabel("MHz")
    axes[2].set_ylabel("W")
    axes[0].set_title("200 ms whole-process telemetry; includes setup/reset/validation")
    for extension in ("png", "svg"):
        fig.savefig(charts / f"telemetry.{extension}", dpi=160)
    plt.close(fig)


def main() -> int:
    """Analyze a finished plan into a new directory, preserving raw evidence.

    Returns:
        Zero for complete validated evidence, even when scenarios are noisy;
        one for invalid/incomplete runs. Stability is a separate result field.

    Raises:
        OSError: If input is missing or output already exists.
        ValueError: If input JSON or required schema is malformed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    manifest = json.loads((directory / "run-manifest.json").read_text())
    by_id = {row["run_id"]: row for row in manifest}
    output = (args.output_directory or directory / "analysis").resolve()
    if not output.is_relative_to(directory):
        raise ValueError("Analysis output must be beneath the evidence directory")
    output.mkdir()
    runs, sample_rows, aggregates = [], [], []
    raw: dict[str, dict[str, Any]] = {}
    signatures: dict[str, Any] = {}
    frozen_hashes: dict[str, str] | None = None
    for repeat in range(1, plan["independent_process_repetitions"] + 1):
        for case in plan["cases_in_fixed_order"]:
            run_id = f"r{repeat:02d}-{case['id']}"
            run = by_id.get(run_id, {"scenario": case})
            path = directory / "runs" / (run_id + ".json")
            data = json.loads(path.read_text()) if path.exists() else {}
            raw[run_id] = data
            errors = audit_run(run, data, plan)
            hashes = data.get("provenance", {}).get("files_sha256", {})
            if frozen_hashes is None:
                frozen_hashes = hashes
            elif hashes != frozen_hashes:
                errors.append("source/extension provenance changed across processes")
            signature = (
                data.get("buffers", {}).get("block_ids"),
                [row.get("object_sha256") for row in data.get("validations", [])],
            )
            if case["id"] in signatures and signature != signatures[case["id"]]:
                errors.append("mapping or data differs across independent repetitions")
            signatures[case["id"]] = signature
            samples = data.get("samples", [])
            summary = {
                "run_id": run_id,
                "scenario": case["id"],
                "repeat": repeat,
                "pid": run.get("pid"),
                "exit_code": run.get("exit_code"),
                "status": "valid" if not errors else "invalid_or_unattempted",
                "errors": errors,
                "recorded": len(samples),
                "failed_recorded": len(samples) if errors else 0,
                "not_produced": max(0, plan["iterations"] - len(samples)),
            }
            for name, key, divisor in (
                ("wall", "cpu_wall_ns", 1e6),
                ("stream", "device_stream_span_ms", 1),
            ):
                values = [
                    sample[key] / divisor
                    for sample in samples
                    if isinstance(sample.get(key), (float, int))
                    and math.isfinite(sample[key])
                    and sample[key] > 0
                ]
                summary[name] = stats(values)
                summary[name]["numeric_values"] = len(values)
            runs.append(summary)
            for index in range(max(plan["iterations"], len(samples))):
                sample = samples[index] if index < len(samples) else {}
                sample_rows.append(
                    {
                        "run_id": run_id,
                        "scenario": case["id"],
                        "repeat": repeat,
                        "pid": run.get("pid"),
                        "slot": index,
                        "status": (summary["status"] if sample else "not_produced"),
                        **sample,
                    }
                )
    thresholds = plan["stability_thresholds_predeclared"]
    for case in plan["cases_in_fixed_order"]:
        group = [run for run in runs if run["scenario"] == case["id"]]
        complete = all(run["status"] == "valid" for run in group)
        aggregate = {
            "scenario": case["id"],
            "complete": complete,
            "failed_recorded": sum(run["failed_recorded"] for run in group),
            "not_produced": sum(run["not_produced"] for run in group),
        }
        reasons = []
        for metric in ("wall", "stream"):
            medians = [run[metric]["median"] for run in group]
            available = all(value is not None for value in medians)
            if not available:
                aggregate[metric] = {"available": False}
                continue
            median = statistics.median(medians)
            spread = (max(medians) - min(medians)) / median
            p95 = max(run[metric]["p95"] / run[metric]["median"] for run in group)
            p99 = max(run[metric]["p99"] / run[metric]["median"] for run in group)
            drift = max(run[metric]["drift"] for run in group)
            aggregate[metric] = {
                "available": True,
                "run_medians_ms": medians,
                "median_of_run_medians_ms": median,
                "median_spread": spread,
                "worst_p95_ratio": p95,
                "worst_p99_ratio": p99,
                "worst_quarter_drift": drift,
            }
            for value, limit, label in (
                (
                    spread,
                    thresholds["between_process_median_spread_fraction_max"],
                    "between-process median spread",
                ),
                (p95, thresholds["worst_p95_over_median_max"], "p95/median"),
                (p99, thresholds["worst_p99_over_median_max"], "p99/median"),
                (
                    drift,
                    thresholds["within_run_first_last_quarter_median_drift_max"],
                    "first/last quarter drift",
                ),
            ):
                if value > limit:
                    reasons.append(f"{metric} {label}: {value:.4f} > {limit}")
        aggregate["reasons"] = reasons
        aggregate["classification"] = (
            "incomplete" if not complete else "noisy" if reasons else "stable_round1"
        )
        aggregates.append(aggregate)
    telemetry = [
        json.loads(line)
        for line in (directory / "telemetry.jsonl").read_text().splitlines()
    ]
    numeric_telemetry: dict[str, Any] = {}
    for field in dict.fromkeys(
        key for row in telemetry for key in row.get("values", {})
    ):
        numeric = []
        for row in telemetry:
            try:
                value = float(row.get("values", {}).get(field, "nan"))
                if math.isfinite(value):
                    numeric.append(value)
            except ValueError:
                pass
        numeric_telemetry[field] = {
            "numeric_rows": len(numeric),
            "unavailable_or_nonnumeric_rows": len(telemetry) - len(numeric),
            "minimum": min(numeric, default=None),
            "maximum": max(numeric, default=None),
        }
    stable = [row for row in aggregates if row["classification"] == "stable_round1"]
    candidates = [row for row in stable if row["scenario"].endswith("-n4")]
    target = max(
        candidates,
        key=lambda row: row["wall"]["median_of_run_medians_ms"],
        default=None,
    )
    result = {
        "measurement_git_sha": plan["git_sha"],
        "runs": runs,
        "scenarios": aggregates,
        "planned_samples": (
            len(plan["cases_in_fixed_order"])
            * plan["independent_process_repetitions"]
            * plan["iterations"]
        ),
        "recorded_samples": sum(run["recorded"] for run in runs),
        "failed_recorded_samples": sum(run["failed_recorded"] for run in runs),
        "not_produced_samples": sum(run["not_produced"] for run in runs),
        "distinct_pids": len({run["pid"] for run in runs if run["pid"] is not None}),
        "all_runs_valid": all(run["status"] == "valid" for run in runs),
        "baseline_ready_for_stable_subset": bool(stable),
        "recommended_profile_group": target["scenario"] if target else None,
        "telemetry_scope": plan["telemetry_scope"],
        "telemetry_rows": len(telemetry),
        "telemetry_ranges": numeric_telemetry,
        "plotting_matplotlib_version": matplotlib.__version__,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_csv(output / "samples.csv", sample_rows)
    write_csv(
        output / "run-summary.csv",
        [
            {
                **{
                    key: value
                    for key, value in run.items()
                    if key not in ("wall", "stream")
                },
                **{
                    f"{metric}_{key}": value
                    for metric in ("wall", "stream")
                    for key, value in run[metric].items()
                },
            }
            for run in runs
        ],
    )
    write_csv(
        output / "telemetry.csv",
        [
            {
                **{key: value for key, value in row.items() if key != "values"},
                **row.get("values", {}),
            }
            for row in telemetry
        ],
    )
    export_figures(output, plan["cases_in_fixed_order"], runs, raw, telemetry)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "all_runs_valid",
                    "recorded_samples",
                    "failed_recorded_samples",
                    "not_produced_samples",
                    "distinct_pids",
                    "recommended_profile_group",
                )
            }
        )
    )
    return 0 if result["all_runs_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
