# SPDX-License-Identifier: Apache-2.0
"""Export phase-aware, correlation-based evidence from a saved GQA nsys trace.

Reads SQLite in read-only mode. Does not initialize CUDA or alter raw data.
Durations across streams are merged as interval unions, never blindly added.
"""

# Standard
from collections import Counter
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import math
import sqlite3
import statistics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save every record, including an explicit empty-file header when empty.

    Args:
        path: New CSV destination.
        rows: Original or derived records without outlier removal.

    Returns:
        None.

    Raises:
        OSError: If destination creation or writing fails.
    """
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["no_records"]
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Compute the union of timestamp intervals on a common trace clock.

    Args:
        intervals: Start/end nanoseconds; overlapping intervals count once.

    Returns:
        Covered elapsed nanoseconds, zero for no intervals.

    Raises:
        ValueError: If an interval ends before it starts.
    """
    if not intervals:
        return 0
    ordered = sorted(intervals)
    if any(end < start for start, end in ordered):
        raise ValueError("Reversed activity interval")
    previous_start, previous_end = ordered[0]
    total = 0
    for start, end in ordered[1:]:
        if end < start:
            raise ValueError("Reversed activity interval")
        if start > previous_end:
            total += previous_end - previous_start
            previous_start, previous_end = start, end
        else:
            previous_end = max(previous_end, end)
    return total + previous_end - previous_start


def describe(values: list[int | float]) -> dict[str, Any]:
    """Summarize all values, retaining negative launch-to-start differences.

    Args:
        values: Nanosecond durations or signed timestamp differences.

    Returns:
        Count, minimum, median, nearest-rank p95 and maximum; None if empty.
    """
    ordered = sorted(values)
    return {
        "count": len(values),
        "minimum_ns": min(values, default=None),
        "median_ns": statistics.median(values) if values else None,
        "p95_ns": ordered[math.ceil(len(values) * 0.95) - 1] if values else None,
        "maximum_ns": max(values, default=None),
    }


def main() -> int:
    """Export raw activities, unique launch joins and completion relationships.

    Returns:
        Zero for structurally matched evidence, one for integrity discrepancies.
        Profiler warnings remain independent limitations even when exit is zero.

    Raises:
        OSError: If evidence cannot be read or output already exists.
        sqlite3.Error: If the installed export schema cannot be queried.
        ValueError: If required NVTX measurement boundaries are missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = (args.output_directory or directory / "analysis").resolve()
    if not output.is_relative_to(directory):
        raise ValueError("Analysis must remain under the evidence directory")
    output.mkdir()
    payload = json.loads((directory / "payload.json").read_text())
    with sqlite3.connect(
        (directory / "store-n4.sqlite").as_uri() + "?mode=ro", uri=True
    ) as connection:
        connection.row_factory = sqlite3.Row
        strings = dict(connection.execute("SELECT id,value FROM StringIds"))
        phases = [dict(row) for row in connection.execute("SELECT * FROM NVTX_EVENTS")]
        for phase in phases:
            phase["label"] = phase["text"] or strings.get(phase["textId"])
        measured = [phase for phase in phases if phase["label"] == "gqa.measurement"]
        if len(measured) != 1:
            raise ValueError("Need exactly one complete measurement NVTX range")
        window = measured[0]
        if window["end"] is None:
            raise ValueError("Unclosed measurement range")

        def phase_for(start: int, end: int) -> str:
            """Name the single enclosing leaf phase, or preserve ambiguity."""
            matches = [
                phase["label"]
                for phase in phases
                if phase["label"] != "gqa.capture"
                and phase["start"] <= start
                and end <= phase["end"]
            ]
            return matches[0] if len(matches) == 1 else "outside_or_crossing"

        runtime = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
            )
        ]
        for row in runtime:
            row["name"] = strings[row["nameId"]]
            row["phase"] = phase_for(row["start"], row["end"])
            row["duration_ns"] = row["end"] - row["start"]
        kernels = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
            )
        ]
        copies = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
            )
        ]
        diagnostics = [
            dict(row) for row in connection.execute("SELECT * FROM DIAGNOSTIC_EVENT")
        ]
        for row in kernels:
            row["name"] = strings[row["demangledName"]]
        for row in kernels + copies:
            row["phase"] = phase_for(row["start"], row["end"])
        sql = """
SELECT k.start AS kernel_start_ns, k.end AS kernel_end_ns,
       k.correlationId, k.globalPid, k.contextId, k.streamId, k.deviceId,
       r.start AS api_start_ns, r.end AS api_end_ns, r.globalTid, s.value AS api,
       n.value AS kernel
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
 ON k.correlationId = r.correlationId
 AND k.globalPid = (r.globalTid & -16777216)
JOIN StringIds AS s ON r.nameId = s.id
JOIN StringIds AS n ON k.demangledName = n.id
WHERE n.value LIKE '%multi_layer_block_transfer_kernel%'
 AND s.value LIKE 'cudaLaunchKernel%'
ORDER BY k.start
"""
        joined = [dict(row) for row in connection.execute(sql)]
    (output / "launch-join.sql").write_text(sql)
    write_csv(output / "phases.csv", phases)
    write_csv(output / "runtime.csv", runtime)
    write_csv(output / "kernels.csv", kernels)
    write_csv(output / "memcopies.csv", copies)
    write_csv(output / "diagnostics.csv", diagnostics)
    measured_runtime = [row for row in runtime if row["phase"] == "gqa.measurement"]
    measured_kernels = [row for row in kernels if row["phase"] == "gqa.measurement"]
    measured_copies = [row for row in copies if row["phase"] == "gqa.measurement"]
    records: list[dict[str, Any]] = []
    issues: list[str] = []
    for row in joined:
        row["phase"] = phase_for(row["kernel_start_ns"], row["kernel_end_ns"])
        row["kernel_duration_ns"] = row["kernel_end_ns"] - row["kernel_start_ns"]
        row["launch_api_duration_ns"] = row["api_end_ns"] - row["api_start_ns"]
        row["launch_start_to_kernel_start_ns"] = (
            row["kernel_start_ns"] - row["api_start_ns"]
        )
        row["launch_end_to_kernel_start_ns_signed"] = (
            row["kernel_start_ns"] - row["api_end_ns"]
        )
        if row["phase"] != "gqa.measurement":
            continue
        index = len(records)
        row["sample_index_by_single_thread_order"] = index
        candidates = [
            item
            for item in measured_runtime
            if item["globalTid"] == row["globalTid"]
            and item["name"].partition("_v")[0] == "cudaEventSynchronize"
            and item["start"] >= row["api_end_ns"]
        ]
        wait = min(candidates, key=lambda item: item["start"], default=None)
        row["wait_start_ns"] = wait["start"] if wait else None
        row["wait_end_ns"] = wait["end"] if wait else None
        row["wait_duration_ns"] = wait["duration_ns"] if wait else None
        row["wait_return_after_kernel_end"] = (
            wait is not None
            and wait["returnValue"] == 0
            and wait["end"] >= row["kernel_end_ns"]
        )
        row["kernel_end_to_wait_return_ns"] = (
            wait["end"] - row["kernel_end_ns"] if wait else None
        )
        prior_records = [
            item
            for item in measured_runtime
            if item["globalTid"] == row["globalTid"]
            and item["name"].startswith("cudaEventRecord")
            and item["end"] <= row["api_start_ns"]
        ]
        start_event = max(prior_records, key=lambda item: item["end"], default=None)
        row["event_record_end_to_launch_start_ns"] = (
            row["api_start_ns"] - start_event["end"] if start_event else None
        )
        queries = [
            item
            for item in measured_runtime
            if start_event
            and item["name"].startswith("cudaPointerGetAttributes")
            and item["globalTid"] == row["globalTid"]
            and start_event["end"] <= item["start"]
            and item["end"] <= row["api_start_ns"]
        ]
        row["pointer_query_count"] = len(queries)
        row["pointer_query_api_union_ns"] = union_ns(
            [(item["start"], item["end"]) for item in queries]
        )
        row["profiled_sample_wall_ns"] = (
            payload["samples"][index]["cpu_wall_ns"]
            if index < len(payload["samples"])
            else None
        )
        records.append(row)
    expected = payload["arguments"]["iterations"]
    if len(joined) != payload["native_calls_executed"]:
        issues.append("full native launch join count mismatch or nonunique joins")
    if len(records) != expected or len(measured_kernels) != expected:
        issues.append("measurement kernel count mismatch")
    if len({(row["globalPid"], row["correlationId"]) for row in joined}) != len(joined):
        issues.append("nonunique process/correlation launch join")
    bad_waits = [row for row in records if not row["wait_return_after_kernel_end"]]
    if bad_waits:
        issues.append(
            f"{len(bad_waits)} kernel completions not chronologically bounded "
            "by successful event wait; no tolerance applied"
        )
    for index, row in enumerate(records[:-1]):
        if (
            row["wait_end_ns"] is None
            or row["wait_end_ns"] > records[index + 1]["api_start_ns"]
        ):
            issues.append("wait crossed next iteration launch")
    if any(row["globalTid"] != window["globalTid"] for row in records):
        issues.append("launch not on measured NVTX thread")
    if not payload["passed"] or len(payload["samples"]) != expected:
        issues.append("benchmark validation or sample count failed")
    write_csv(output / "native-launches-all.csv", joined)
    write_csv(output / "measured-associations.csv", records)
    api_summary = []
    for name, count in sorted(Counter(row["name"] for row in measured_runtime).items()):
        rows = [row for row in measured_runtime if row["name"] == name]
        api_summary.append(
            {
                "name": name,
                "count": count,
                "time_ns": sum(row["duration_ns"] for row in rows),
                "return_values": dict(Counter(row["returnValue"] for row in rows)),
            }
        )
    write_csv(output / "measured-api-summary.csv", api_summary)
    summaries = {
        key: describe([row[key] for row in records if row[key] is not None])
        for key in [
            "kernel_duration_ns",
            "launch_api_duration_ns",
            "launch_start_to_kernel_start_ns",
            "launch_end_to_kernel_start_ns_signed",
            "wait_duration_ns",
            "kernel_end_to_wait_return_ns",
            "event_record_end_to_launch_start_ns",
            "pointer_query_api_union_ns",
        ]
    }
    phase_counts = []
    for phase in phases:
        name = phase["label"]
        phase_counts.append(
            {
                "phase": name,
                "start_ns": phase["start"],
                "end_ns": phase["end"],
                "duration_ns": phase["end"] - phase["start"],
                "kernels": sum(
                    phase["start"] <= row["start"] and row["end"] <= phase["end"]
                    for row in kernels
                ),
                "memcopies": sum(
                    phase["start"] <= row["start"] and row["end"] <= phase["end"]
                    for row in copies
                ),
            }
        )
    result = {
        "structural_analysis_ok": not issues,
        "issues": issues,
        "scope": (
            "single native process, prepared store batch; not RPC or engine pipeline"
        ),
        "pid": payload["pid"],
        "native_tid": payload["native_tid"],
        "measurement": {
            "start_ns": window["start"],
            "end_ns": window["end"],
            "duration_ns": window["end"] - window["start"],
        },
        "phases": phase_counts,
        "phase_counts_include_parent_ranges_do_not_sum": True,
        "native_kernels_full": len(joined),
        "native_kernels_measured": len(records),
        "measured_memcopies": len(measured_copies),
        "streams": sorted(
            {
                (row["deviceId"], row["contextId"], row["streamId"])
                for row in measured_kernels
            }
        ),
        "measured_gpu_activity_union_ns": union_ns(
            [(row["start"], row["end"]) for row in measured_kernels + measured_copies]
        ),
        "measured_kernel_interval_union_ns": union_ns(
            [(row["start"], row["end"]) for row in measured_kernels]
        ),
        "measured_api_summary": api_summary,
        "per_call_statistics": summaries,
        "measured_pointer_queries_total": sum(
            row["pointer_query_count"] for row in records
        ),
        "negative_launch_end_to_start_count": sum(
            row["launch_end_to_kernel_start_ns_signed"] < 0 for row in records
        ),
        "completion_time_inconsistencies": [
            {
                key: row[key]
                for key in (
                    "sample_index_by_single_thread_order",
                    "correlationId",
                    "kernel_end_to_wait_return_ns",
                )
            }
            for row in bad_waits
        ],
        "sample_association": (
            "one launch/call, single thread and stream; ordered association, "
            "no per-sample NVTX ID"
        ),
        "profiler_warnings": [row for row in diagnostics if row["severity"] >= 2],
        "trace_completeness": (
            "limited: preserve profiler warnings, do not infer no dropped events"
        ),
        "cpu_descriptor_prepare_time": (
            "unavailable: no CPU function range or stack samples"
        ),
        "critical_path": (
            "partial ordered dependency evidence only; "
            "no complete scheduler/event graph"
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
