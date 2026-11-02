# SPDX-License-Identifier: Apache-2.0
"""Run a predeclared GQA baseline matrix in serial independent GPU processes."""

# Standard
from pathlib import Path
from typing import Any, TextIO
import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "build/kv-transfer-lab"
FIELDS = (
    "timestamp",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.current.graphics",
    "power.draw",
    "power.limit",
    "utilization.gpu",
    "utilization.memory",
    "pstate",
    "memory.used",
    "memory.total",
)


def read_telemetry(stream: TextIO, output: Path, state: dict[str, Any]) -> None:
    """Preserve every NVML CSV line with receiver time and a coarse process tag.

    Args:
        stream: Owned nvidia-smi stdout.
        output: Exclusive JSONL destination.
        state: Current process tag; this is not a measured-phase annotation.

    Returns:
        None. Retains N/A strings and malformed/event lines without filtering.
    """
    with output.open("x") as log:
        for line in stream:
            row = next(csv.reader([line]))
            record = {
                "received_unix_ns": time.time_ns(),
                "received_monotonic_ns": time.monotonic_ns(),
                "process_tag": state.get("run_id"),
                "raw": line.rstrip(),
            }
            if len(row) == len(FIELDS):
                record["values"] = dict(
                    zip(FIELDS, [value.strip() for value in row], strict=True)
                )
            else:
                record["parse_status"] = "message_or_unparsed; raw retained"
            log.write(json.dumps(record) + "\n")
            log.flush()


def save(path: Path, data: Any) -> None:
    """Save controller state after each completed process.

    Args:
        path: Controller-owned state file.
        data: JSON-serializable record.

    Returns:
        None.
    """
    path.write_text(json.dumps(data, indent=2))


def main() -> int:
    """Execute the fixed matrix, preserving failures and all raw benchmark files.

    Returns:
        Zero only if all planned processes finish successfully and pass validation.

    Raises:
        RuntimeError: If checkout identity, telemetry startup or idle checks fail.
        OSError: If evidence creation or process startup fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.directory.resolve()
    if not out.is_relative_to(LAB):
        raise RuntimeError("Evidence directory must be beneath the lab")
    plan = json.loads((out / "plan.json").read_text())
    benchmark = ROOT / plan["benchmark"]
    if hashlib.sha256(benchmark.read_bytes()).hexdigest() != plan["script_sha256"]:
        raise RuntimeError("Benchmark changed after the plan was registered")
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != plan["git_sha"]:
        raise RuntimeError("Checkout changed after the plan was registered")
    with urllib.request.urlopen("http://127.0.0.1:8080/status", timeout=3) as response:
        service = json.load(response)
    if (
        service["active_sessions"]
        or service["storage_manager"]["store_controller"]["in_flight_task_count"]
    ):
        raise RuntimeError("Existing service is busy")
    environment = os.environ.copy()
    environment.update(plan["environment"])
    cache = LAB / "runtime-cache"
    environment.update(
        {
            "XDG_CACHE_HOME": str(cache),
            "TORCH_HOME": str(cache / "torch"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "NUMBA_CACHE_DIR": str(cache / "numba"),
        }
    )
    python = str(LAB / ".venv/bin/python")
    probe = (
        "import json,torch; print(json.dumps({'torch':str(torch.__version__),"
        "'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),"
        "'free_total_gpu':torch.cuda.mem_get_info(),'device':torch.cuda.get_device_name(0)}))"
    )
    actual = subprocess.check_output([python, "-c", probe], env=environment, text=True)
    save(
        out / "runtime-preflight.json",
        {
            "probe": json.loads(actual),
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "fixed_environment": plan["environment"],
            "existing_service": service,
        },
    )
    snapshot = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (out / "nvidia-smi-before.log").write_text(snapshot.stdout + snapshot.stderr)
    telemetry_command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=" + ",".join(FIELDS),
        "--format=csv,noheader,nounits",
        "--loop-ms=" + str(plan["telemetry_interval_ms"]),
    ]
    runs: list[dict[str, Any]] = []
    state: dict[str, Any] = {"run_id": None}
    active: subprocess.Popen[str] | None = None
    with (out / "telemetry-stderr.log").open("x") as stderr:
        monitor = subprocess.Popen(
            telemetry_command, stdout=subprocess.PIPE, stderr=stderr, text=True
        )
        if monitor.stdout is None:
            raise RuntimeError("Telemetry stdout unavailable")
        reader = threading.Thread(
            target=read_telemetry,
            args=(monitor.stdout, out / "telemetry.jsonl", state),
            daemon=True,
        )
        reader.start()
        stopped_early = False
        try:
            time.sleep(0.3)
            if monitor.poll() is not None:
                raise RuntimeError("Telemetry startup failed; inspect stderr")
            for repeat in range(1, plan["independent_process_repetitions"] + 1):
                for case in plan["cases_in_fixed_order"]:
                    run_id = f"r{repeat:02d}-{case['id']}"
                    result_file = out / "runs" / (run_id + ".json")
                    argv = [python, str(benchmark)]
                    geometry = {
                        **plan["geometry"],
                        "direction": case["direction"],
                        "mapping": case["mapping"],
                        "num_objects": case["num_objects"],
                        "warmup": plan["warmup"],
                        "iterations": plan["iterations"],
                        "seed": plan["seed"],
                        "output": str(result_file),
                    }
                    for key, value in geometry.items():
                        argv.extend(["--" + key.replace("_", "-"), str(value)])
                    record = {
                        "run_id": run_id,
                        "repeat": repeat,
                        "scenario": case,
                        "command": argv,
                        "expected_samples": plan["iterations"],
                        "start_unix_ns": time.time_ns(),
                        "start_monotonic_ns": time.monotonic_ns(),
                        "loadavg_before": Path("/proc/loadavg").read_text().strip(),
                        "result_path": str(result_file),
                        "timeout": False,
                        "forced_kill": False,
                    }
                    state["run_id"] = run_id
                    with (out / "runs" / (run_id + ".log")).open("x") as log:
                        active = subprocess.Popen(
                            argv,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                        record["pid"] = active.pid
                        try:
                            code = active.wait(timeout=180)
                        except subprocess.TimeoutExpired:
                            record["timeout"] = True
                            active.send_signal(signal.SIGINT)
                            try:
                                code = active.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                active.kill()
                                code = active.wait(timeout=10)
                                record["forced_kill"] = True
                    record.update(
                        {
                            "exit_code": code,
                            "end_unix_ns": time.time_ns(),
                            "end_monotonic_ns": time.monotonic_ns(),
                            "loadavg_after": Path("/proc/loadavg").read_text().strip(),
                        }
                    )
                    state["run_id"] = None
                    try:
                        result = json.loads(result_file.read_text())
                        record["reported_passed"] = result.get("passed", False)
                        record["recorded_samples"] = len(result.get("samples", []))
                    except (OSError, ValueError) as error:
                        record["reported_passed"] = False
                        record["recorded_samples"] = 0
                        record["result_read_error"] = str(error)
                    runs.append(record)
                    save(out / "run-manifest.json", runs)
                    print(
                        json.dumps(
                            {
                                key: record[key]
                                for key in (
                                    "run_id",
                                    "pid",
                                    "exit_code",
                                    "reported_passed",
                                    "recorded_samples",
                                )
                            }
                        ),
                        flush=True,
                    )
                    active = None
                    if code != 0 or not record["reported_passed"]:
                        stopped_early = True
                        break
                    time.sleep(plan["pause_between_runs_seconds"])
                if stopped_early:
                    break
        finally:
            if active is not None and active.poll() is None:
                active.send_signal(signal.SIGINT)
                try:
                    active.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    active.kill()
                    active.wait(timeout=10)
            state["run_id"] = None
            if monitor.poll() is None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait(timeout=10)
            reader.join(timeout=10)
            save(
                out / "telemetry-command.json",
                {
                    "command": telemetry_command,
                    "exit_code": monitor.returncode,
                    "intentional_stop": True,
                    "reader_still_alive": reader.is_alive(),
                    "interval_ms": plan["telemetry_interval_ms"],
                },
            )
    expected = (
        len(plan["cases_in_fixed_order"]) * plan["independent_process_repetitions"]
    )
    success = len(runs) == expected and all(
        run["exit_code"] == 0 and run["reported_passed"] for run in runs
    )
    save(
        out / "controller-result.json",
        {
            "passed": success,
            "planned_processes": expected,
            "attempted_processes": len(runs),
            "stopped_early": stopped_early,
            "sampling_scope": plan["scope"],
        },
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
