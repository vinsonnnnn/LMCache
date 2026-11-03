# SPDX-License-Identifier: Apache-2.0
"""Profile the benchmark's public helpers with coarse experiment-only NVTX ranges.

Accepts the same arguments as bench_gqa_transfer.py. Run under nsys with
--capture-range=cudaProfilerApi --capture-range-end=stop. No per-call markers,
kernel copies, monkeypatches or production changes are used. Timing from this
instrumented run must not replace the unprofiled baseline.
"""

# Standard
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import threading
import traceback

# Third Party
import torch
import torch.cuda.profiler

LAB = Path(__file__).resolve().parents[1]


def load_benchmark() -> ModuleType:
    """Load the unchanged local benchmark without invoking its CLI main.

    Returns:
        Module exposing public prepare, transfer, validate and provenance helpers.

    Raises:
        ImportError: If the local benchmark cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(
        "gqa_profile_benchmark", LAB / "scripts/bench_gqa_transfer.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load benchmark")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_phases(bench: ModuleType, args: Any, result: dict[str, Any]) -> None:
    """Call original helpers, keeping reset, completion and validation intact.

    Args:
        bench: Unchanged benchmark module with public helpers.
        args: Its validated argparse namespace.
        result: Partial evidence retained if a phase fails.

    Returns:
        None. Marks passed only after exact boundary validation.

    Raises:
        Exception: Propagates the original allocation, transfer or data failure.
    """
    result["provenance"] = bench.provenance(args)
    capture = nullcontext() if args.correctness_only else torch.cuda.profiler.profile()
    plan = None
    try:
        with capture, torch.cuda.nvtx.range("gqa.capture"):
            with torch.cuda.nvtx.range("gqa.prepare"):
                plan = bench.prepare(args, result["provenance"])
            result["buffers"] = {
                "gpu_shapes": [list(t.shape) for t in plan.gpu],
                "gpu_strides": [list(t.stride()) for t in plan.gpu],
                "host_shapes": [list(t.shape) for t in plan.host],
                "host_strides": [list(t.stride()) for t in plan.host],
                "all_host_objects_pinned": all(t.is_pinned() for t in plan.host),
                "block_ids": plan.ids,
                "producer_handle": int(plan.producer.cuda_stream),
                "transfer_handle": int(plan.stream.cuda_stream),
                "stream_handles_are_not_cupti_stream_ids": True,
                "layout": "NL x [2,NB,BS,NH,HS]; host [2,L,C,NH*HS]",
            }
            with torch.cuda.nvtx.range("gqa.precheck"):
                bench.transfer(plan, measured=False)
                result["validations"].append(bench.validate(plan, "precheck"))
            if not args.correctness_only:
                with torch.cuda.nvtx.range("gqa.warmup"):
                    for _ in range(args.warmup):
                        bench.transfer(plan, measured=False)
                with torch.cuda.nvtx.range("gqa.validate_warmup"):
                    result["validations"].append(bench.validate(plan, "after_warmup"))
                with torch.cuda.nvtx.range("gqa.measurement"):
                    for index in range(args.iterations):
                        sample = bench.transfer(plan, measured=True)
                        assert sample is not None and sample["completion_event_ready"]
                        sample.update(
                            index=index,
                            effective_bytes=result["provenance"][
                                "effective_bytes_per_call"
                            ],
                        )
                        result["samples"].append(sample)
                with torch.cuda.nvtx.range("gqa.validate_measurement"):
                    result["validations"].append(
                        bench.validate(plan, "after_measurement")
                    )
            result["native_calls_executed"] = plan.call_count
            result["destination_resets_executed"] = plan.reset_count
            result["actual_path"] = "native multi_layer_block_kv_transfer block kernel"
        result["passed"] = True
    except BaseException:
        if plan is not None:
            try:
                plan.stream.synchronize()
            except Exception as error:
                result["cleanup_error"] = str(error)
        raise


def main() -> int:
    """Serialize the GPU run and preserve its output, including first failures.

    Returns:
        Zero after validation, one on failure, or 130 on interruption.

    Raises:
        SystemExit: For the benchmark's argument errors or help.
    """
    bench = load_benchmark()
    args = bench.parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.output.open("x")
    except OSError as error:
        print(str(error), file=sys.stderr)
        return 1
    result: dict[str, Any] = {
        "passed": False,
        "arguments": {**vars(args), "output": str(args.output.resolve())},
        "pid": os.getpid(),
        "native_tid": threading.get_native_id(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "original prepared native batch helpers; no RPC; profiled run",
        "capture": "CUDA profiler API around coarse phases; disabled for tiny check",
        "stage_times_ms": {
            key: "unavailable" for key in ("cpu_prepare", "kernel", "dma")
        },
        "samples": [],
        "validations": [],
    }
    code = 1
    try:
        with (LAB / "gpu-experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not __debug__:
                raise RuntimeError("Correctness assertions must remain enabled")
            run_phases(bench, args, result)
            code = 0
    except KeyboardInterrupt:
        result["error"] = traceback.format_exc()
        code = 130
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], file=sys.stderr)
    finally:
        with output:
            json.dump(result, output, indent=2, allow_nan=False)
        print(
            json.dumps({"passed": result["passed"], "samples": len(result["samples"])})
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
