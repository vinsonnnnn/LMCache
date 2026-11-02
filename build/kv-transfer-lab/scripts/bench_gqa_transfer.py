# SPDX-License-Identifier: Apache-2.0
"""Benchmark a prepared native GQA block-transfer batch with completion.

Uses the public LMCache block-transfer entry used by engine-driven gather/scatter
and tests/v1/test_mp_mem_kernels.py. No RPC, cache keys, allocation, pinning or
mapping preparation is timed. This is neither pure kernel timing nor a complete
engine/server operation. CUDA events cover the whole native batch's stream span,
including possible host-submission gaps; individual kernel/DMA times are unknown.

Only --path auto is supported: this checkout has no safe public direct/kernel
selector for this entry. Native identity and audited source establish the path;
no profiler kernel-name verification is claimed. Buffers and source data are
reused (hot-buffer experiment); the destination is poisoned before EVERY call.
There is no cache/dedup layer to turn repeated operations into hits or no-ops.

Outputs must stay under build/kv-transfer-lab and are never overwritten.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
import argparse
import fcntl
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback

# Third Party
import torch

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "build/kv-transfer-lab"
SENTINEL = -9.0
GUARD_ELEMENTS = 16
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_arguments() -> argparse.Namespace:
    """Parse and validate experiment controls.

    Returns:
        Arguments for one direction and one fixed NHD geometry.

    Raises:
        SystemExit: If arguments are invalid or help is requested.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name, default in (
        ("layers", 28),
        ("kv-heads", 2),
        ("head-dim", 128),
        ("block-size", 16),
        ("chunk-size", 256),
        ("num-objects", 4),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--direction", choices=("store", "retrieve"), required=True)
    parser.add_argument("--mapping", choices=("contiguous", "random"), default="random")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bfloat16")
    parser.add_argument(
        "--path",
        choices=("auto",),
        default="auto",
        help="Audited public native entry; no direct/kernel override exists",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="One untimed transfer plus exact checks; no warmup or samples",
    )
    args = parser.parse_args()
    for name in (
        "layers",
        "kv_heads",
        "head_dim",
        "block_size",
        "chunk_size",
        "num_objects",
    ):
        if not 0 < getattr(args, name) < 2**31:
            parser.error(f"{name} must be a positive 32-bit integer")
    if args.kv_heads > 32 or args.chunk_size % args.block_size:
        parser.error("kv-heads must be <=32 and chunk-size divisible by block-size")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be >=0 and iterations >=1")
    if args.head_dim * DTYPES[args.dtype].itemsize % 2:
        parser.error("head-dim * element-size must be divisible by 2")
    if not args.output.resolve().is_relative_to(LAB):
        parser.error("output must be beneath build/kv-transfer-lab")
    return args


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Check CUDA/native identity and record hardware, source and memory budgets.

    Args:
        args: Validated experiment arguments.

    Returns:
        JSON-serializable provenance and pre-allocation budget evidence.

    Raises:
        RuntimeError: If the project environment, GPU or native backend is missing.
        MemoryError: If conservative GPU/main-memory guards cannot be satisfied.
        ValueError: If the descriptor dimensions exceed their native bounds.
    """
    # First Party
    from lmcache import device_ops, lmcache_native
    import lmcache
    import lmcache.cuda_ops as cuda_ops

    if Path(sys.prefix).resolve() != LAB / ".venv":
        raise RuntimeError("Use build/kv-transfer-lab/.venv/bin/python")
    if Path(lmcache.__file__).resolve() != ROOT / "lmcache/__init__.py":
        raise RuntimeError("LMCache must come from this checkout")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A real NVIDIA CUDA device is required")
    torch.cuda.set_device(0)
    device_ops.ensure_native()
    if (
        device_ops.multi_layer_block_kv_transfer
        is not cuda_ops.multi_layer_block_kv_transfer
    ):
        raise RuntimeError(
            "Native binding identity failed; torch fallback is not allowed"
        )
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This device does not support BF16")
    blocks = args.num_objects * (args.chunk_size // args.block_size)
    pool_blocks = 2 * blocks + 2
    if pool_blocks >= 2**31 or blocks >= 2**31:
        raise ValueError("Block count exceeds the native descriptor range")
    token_bytes = (
        2 * args.layers * args.kv_heads * args.head_dim * DTYPES[args.dtype].itemsize
    )
    payload = args.num_objects * args.chunk_size * token_bytes
    pool = pool_blocks * args.block_size * token_bytes
    guards = 2 * GUARD_ELEMENTS * DTYPES[args.dtype].itemsize * args.num_objects
    gpu_estimate = pool + (blocks + args.layers) * 8
    cpu_estimate = pool + 2 * payload + guards + pool // args.layers
    free, total = torch.cuda.mem_get_info()
    memory = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    ram = int(memory["MemAvailable"].split()[0]) * 1024
    if free < gpu_estimate + 1024**3 or ram < 2 * cpu_estimate + 512 * 1024**2:
        raise MemoryError(
            "Insufficient GPU/RAM headroom for buffers and exact references"
        )
    paths = [
        Path(__file__),
        Path(cuda_ops.__file__),
        Path(lmcache_native.__file__),
        ROOT / "csrc/cuda/mp_mem_kernels.cu",
        ROOT / "csrc/cuda/mp_mem_kernels.cuh",
        ROOT / "tests/v1/test_mp_mem_kernels.py",
        LAB / "path-audit.md",
    ]
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()
    }
    properties = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        driver = "unavailable: nvidia-smi query failed"
    return {
        "python": sys.executable,
        "torch": str(torch.__version__),
        "torch_cuda_build_version": torch.version.cuda,
        "driver_version_nvidia_smi": driver,
        "ignored_lab_script_hashed_separately": True,
        "cuda_runtime_package_version": version("nvidia-cuda-runtime"),
        "cuda_runtime_version": "unavailable: not queried",
        "cuda_toolkit_version": "unavailable: compiler not queried",
        "device": properties.name,
        "device_index": 0,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_free_before_allocation_bytes": free,
        "gpu_total_bytes": total,
        "ram_available_before_allocation_bytes": ram,
        "git_sha": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty_status": subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain=v1"], text=True
        ),
        "extension_path": cuda_ops.__file__,
        "files_sha256": hashes,
        "source_audit_note": (
            "path-audit.md is frozen at eaceddaa; current native "
            "host-alias fix is included"
        ),
        "payload_bytes_per_token": token_bytes,
        "effective_bytes_per_call": payload,
        "num_selected_blocks": blocks,
        "pool_blocks": pool_blocks,
        "gpu_buffer_estimate_bytes": gpu_estimate,
        "cpu_buffer_estimate_bytes": cpu_estimate,
        "gpu_headroom_required_bytes": 1024**3,
        "cpu_headroom_rule": "2 * CPU estimate + 512 MiB",
    }


def prepare(args: argparse.Namespace, metadata: dict[str, Any]) -> PreparedTransfer:
    """Allocate guarded pinned buffers and NHD tensors, and prebuild all batches.

    Args:
        args: Validated geometry, direction and mapping controls.
        metadata: Provenance containing the checked pool/block sizes.

    Returns:
        Owned buffers/references, native argument tuples and initialized events.

    Raises:
        RuntimeError: If pinned allocation, CUDA allocation or native setup fails.
        AssertionError: If mapping, contiguity or pinned-memory invariants fail.
    """
    # First Party
    from lmcache import device_ops
    import lmcache.lmcache_native as native

    dtype = DTYPES[args.dtype]
    bpc = args.chunk_size // args.block_size
    count = metadata["num_selected_blocks"]
    nb = metadata["pool_blocks"]
    ids = (
        list(range(1, count + 1))
        if args.mapping == "contiguous"
        else random.Random(args.seed).sample(range(1, nb - 1), count)
    )
    assert len(set(ids)) == count and all(0 <= i < nb for i in ids)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (2, nb, args.block_size, args.kv_heads, args.head_dim)
    gpu_reference = [
        torch.full(shape, SENTINEL, dtype=dtype) for _ in range(args.layers)
    ]
    if args.direction == "store":
        for tensor in gpu_reference:
            selected_source = torch.empty(
                (2, count, args.block_size, args.kv_heads, args.head_dim), dtype=dtype
            ).uniform_(-0.5, 0.5, generator=rng)
            tensor[:, ids] = selected_source
    gpu = [tensor.to("cuda:0") for tensor in gpu_reference]
    object_shape = (2, args.layers, args.chunk_size, args.kv_heads * args.head_dim)
    elements = math.prod(object_shape)
    backing = [
        torch.full(
            (elements + 2 * GUARD_ELEMENTS,), SENTINEL, dtype=dtype, pin_memory=True
        )
        for _ in range(args.num_objects)
    ]
    host = [
        tensor.narrow(0, GUARD_ELEMENTS, elements).view(object_shape)
        for tensor in backing
    ]
    expected = [torch.empty(object_shape, dtype=dtype) for _ in host]
    for index, (view, reference) in enumerate(zip(host, expected, strict=True)):
        assert view.is_pinned() and view.is_contiguous()
        selected = ids[index * bpc : (index + 1) * bpc]
        if args.direction == "store":
            for layer in range(args.layers):
                reference[:, layer].copy_(
                    gpu_reference[layer][:, selected].reshape(
                        2, args.chunk_size, args.kv_heads * args.head_dim
                    )
                )
        else:
            view.uniform_(-0.5, 0.5, generator=rng)
            reference.copy_(view)
            for layer in range(args.layers):
                gpu_reference[layer][:, selected] = reference[:, layer].reshape(
                    2, bpc, args.block_size, args.kv_heads, args.head_dim
                )
    descriptor = native.PageBufferShapeDesc()
    for name, value in {
        "kv_size": 2,
        "nl": args.layers,
        "nb": nb,
        "bs": args.block_size,
        "nh": args.kv_heads,
        "hs": args.head_dim,
        "element_size": dtype.itemsize,
    }.items():
        setattr(descriptor, name, value)
    pointers = torch.tensor(
        [t.data_ptr() for t in gpu], dtype=torch.int64, device="cuda:0"
    )
    ids_gpu = torch.tensor(ids, dtype=torch.int64, device="cuda:0")
    direction = (
        native.TransferDirection.D2H
        if args.direction == "store"
        else native.TransferDirection.H2D
    )
    batches = tuple(
        (
            pointers,
            [t.data_ptr() for t in host[i : i + 4]],
            ids_gpu[i * bpc : min(i + 4, args.num_objects) * bpc],
            torch.device("cuda:0"),
            direction,
            descriptor,
            args.chunk_size,
            native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            0,
        )
        for i in range(0, args.num_objects, 4)
    )
    producer = torch.cuda.current_stream(0)
    stream = torch.cuda.Stream(device=0)
    ready = torch.cuda.Event()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    ready.record(producer)
    stream.wait_event(ready)
    with torch.cuda.stream(stream):
        start.record(stream)
        end.record(stream)
    end.synchronize()  # Event allocation and initial dependencies stay outside timing.
    return PreparedTransfer(
        gpu,
        gpu_reference,
        backing,
        host,
        expected,
        ids,
        ids_gpu,
        pointers,
        batches,
        device_ops.multi_layer_block_kv_transfer,
        producer,
        stream,
        ready,
        start,
        end,
        args.direction,
    )


def reset_destination(plan: PreparedTransfer) -> None:
    """Poison every destination payload, preserving unselected sentinel evidence.

    Args:
        plan: Prepared transfer whose preceding call has completed.

    Returns:
        None. Completes resets and preserves an explicit producer->transfer wait.

    Raises:
        RuntimeError: If reset, event recording or synchronization fails.
    """
    with torch.cuda.stream(plan.producer):
        if plan.direction == "store":
            for view in plan.host:
                view.fill_(SENTINEL)
        else:
            for tensor in plan.gpu:
                tensor.index_fill_(1, plan.ids_gpu, SENTINEL)
        plan.ready.record(plan.producer)
    plan.stream.wait_event(plan.ready)
    plan.stream.synchronize()  # Exclude reset and its dependency wait from timing.
    plan.reset_count += 1


def transfer(plan: PreparedTransfer, measured: bool) -> dict[str, Any] | None:
    """Call every prepared native batch and wait for the actual stream to finish.

    Args:
        plan: Buffers and descriptors retained until event completion.
        measured: Whether to record a wall/device sample.

    Returns:
        A sample for measured calls; None for untimed correctness/warmup calls.

    Raises:
        RuntimeError: If native transfer or CUDA completion fails.
    """
    reset_destination(plan)
    with torch.cuda.stream(plan.stream):
        before = time.perf_counter_ns() if measured else 0
        plan.start.record(plan.stream)
        for batch in plan.batches:
            plan.native(*batch)
        plan.end.record(plan.stream)
        plan.end.synchronize()
        wall_ns = time.perf_counter_ns() - before if measured else 0
    plan.call_count += len(plan.batches)
    if not measured:
        return None
    return {
        "cpu_wall_ns": wall_ns,
        "device_stream_span_ms": plan.start.elapsed_time(plan.end),
        "completion_event_ready": plan.end.query(),
        "native_calls": len(plan.batches),
        "destination_reset_executed": True,
    }


def validate(plan: PreparedTransfer, phase: str) -> dict[str, Any]:
    """Compare full GPU pools, host payloads and host guards against CPU references.

    Args:
        plan: Completed transfer and independently prepared CPU references.
        phase: Validation boundary to include in output.

    Returns:
        Exact comparison evidence including payload hashes and untouched regions.

    Raises:
        AssertionError: If any payload, GPU source/untouched block or guard differs.
        RuntimeError: If GPU readback fails.
    """
    for actual, reference in zip(plan.gpu, plan.gpu_reference, strict=True):
        assert torch.equal(actual.cpu(), reference), f"{phase}: GPU pool mismatch"
    hashes = []
    for backing, view, reference in zip(
        plan.host_backing, plan.host, plan.host_reference, strict=True
    ):
        assert torch.equal(view, reference), f"{phase}: host payload mismatch"
        assert bool(torch.all(backing[:GUARD_ELEMENTS] == SENTINEL))
        assert bool(torch.all(backing[-GUARD_ELEMENTS:] == SENTINEL))
        hashes.append(
            hashlib.sha256(view.view(torch.uint8).numpy().tobytes()).hexdigest()
        )
    return {
        "phase": phase,
        "all_gpu_layers_exact": True,
        "host_objects_exact": True,
        "unselected_gpu_blocks_unchanged": True,
        "host_guards_unchanged": True,
        "source_unchanged": True,
        "object_sha256": hashes,
    }


def execute(args: argparse.Namespace, result: dict[str, Any]) -> None:
    """Run precheck, optional warmup/measurement and mandatory boundary validation.

    Args:
        args: Experiment controls.
        result: Mutable JSON report, retaining partial evidence on failure.

    Returns:
        None.

    Raises:
        Exception: Propagates the first setup, CUDA or correctness failure.
    """
    result["phase"] = "provenance"
    result["provenance"] = provenance(args)
    result["phase"] = "prepare"
    plan = prepare(args, result["provenance"])
    try:
        result["buffers"] = {
            "gpu_shapes": [list(t.shape) for t in plan.gpu],
            "gpu_strides": [list(t.stride()) for t in plan.gpu],
            "host_shapes": [list(t.shape) for t in plan.host],
            "host_strides": [list(t.stride()) for t in plan.host],
            "host_storage_offsets": [t.storage_offset() for t in plan.host],
            "all_host_objects_pinned": all(t.is_pinned() for t in plan.host),
            "block_ids": plan.ids,
            "block_ids_shape": list(plan.ids_gpu.shape),
            "block_ids_stride": list(plan.ids_gpu.stride()),
            "pointer_table_shape": list(plan.pointers_gpu.shape),
            "streams": {
                "producer_handle": int(plan.producer.cuda_stream),
                "transfer_handle": int(plan.stream.cuda_stream),
                "dependency": (
                    "producer ready event -> transfer stream wait before timing"
                ),
                "timing_events": (
                    "start/end both recorded on the native transfer stream"
                ),
            },
            "unique_scatter_targets": len(set(plan.ids)),
            "unselected_gpu_blocks": result["provenance"]["pool_blocks"]
            - len(plan.ids),
            "sentinel": SENTINEL,
            "layout": "NL x [2,NB,BS,NH,HS] (NHD); host [2,L,C,NH*HS]",
        }
        result["actual_path"] = (
            "native block-transfer attempted; completion unconfirmed"
        )
        result["phase"] = "precheck"
        transfer(plan, measured=False)
        result["actual_path"] = (
            "LMCache native multi_layer_block_kv_transfer block kernel"
        )
        result["validations"].append(
            validate(plan, "correctness_only" if args.correctness_only else "precheck")
        )
        if not args.correctness_only:
            result["phase"] = "warmup"
            for _ in range(args.warmup):
                transfer(plan, measured=False)
            result["validations"].append(validate(plan, "after_warmup"))
            result["phase"] = "measurement"
            for index in range(args.iterations):
                sample = transfer(plan, measured=True)
                assert sample is not None and sample["completion_event_ready"]
                sample.update(
                    {
                        "index": index,
                        "effective_bytes": result["provenance"][
                            "effective_bytes_per_call"
                        ],
                    }
                )
                result["samples"].append(sample)
            result["validations"].append(validate(plan, "after_measurement"))
        result["native_calls_executed"] = plan.call_count
        result["destination_resets_executed"] = plan.reset_count
        result["warmup_calls_executed"] = 0 if args.correctness_only else args.warmup
        result["measured_calls_executed"] = len(result["samples"])
        result["phase"] = "complete"
        result["passed"] = True
    except BaseException:
        # Retain GPU and pinned owners while retiring any partially queued work.
        # Preserve the first failure if the CUDA context is already unusable.
        try:
            plan.stream.synchronize()
        except Exception as cleanup_error:
            result["cleanup_error"] = str(cleanup_error)
        raise


def main() -> int:
    """Serialize GPU experiments and write success or full first-error evidence.

    Returns:
        Zero only after all required validations pass; nonzero on any failure.

    Raises:
        SystemExit: For argparse errors or help.
    """
    args = parse_arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output = args.output.open("x")
    except OSError as error:
        print(f"Output refused: {error}", file=sys.stderr)
        return 1
    result: dict[str, Any] = {
        "schema_version": 1,
        "passed": False,
        "arguments": {**vars(args), "output": str(args.output.resolve())},
        "actual_path": "not executed",
        "measurement_scope": (
            "prepared native transfer batch + completion; no RPC or full "
            "engine operation"
        ),
        "path_comparison": (
            "unavailable: no public direct/kernel override for this entry"
        ),
        "stage_times_ms": {
            name: "unavailable" for name in ("kernel", "dma", "cpu_prepare", "rpc")
        },
        "device_timing_scope": (
            "entire transfer-stream event interval, may include host enqueue gaps"
        ),
        "wall_timing_scope": (
            "start-event record, native submissions, end-event record and "
            "end-event synchronize"
        ),
        "excluded": [
            "allocation",
            "pinning",
            "Python pointer tables and argument tuples",
            "mapping",
            "references",
            "reset",
            "validation",
            "hashing",
        ],
        "reuse": (
            "fixed source/pinned/GPU buffers; destination poisoned every "
            "call; no cache/dedup; hot buffers"
        ),
        "native_internal_preparation": (
            "pointer queries and native descriptors included"
        ),
        "clock": "perf_counter_ns; CUDA timing events on the actual native stream",
        "cpu_minus_device_interpretation": (
            "unavailable; difference is not identified as Python overhead"
        ),
        "validations": [],
        "samples": [],
    }
    code = 1
    try:
        with (LAB / "gpu-experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not __debug__:
                raise RuntimeError(
                    "Do not disable correctness assertions with python -O"
                )
            execute(args, result)
            code = 0
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], file=sys.stderr)
    except KeyboardInterrupt:
        result["error"] = "Interrupted"
        code = 130
    finally:
        with output:
            json.dump(result, output, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "output": str(args.output),
                    "samples": len(result["samples"]),
                }
            ),
            flush=True,
        )
    return code


@dataclass
class PreparedTransfer:
    """Own buffers, immutable CPU references, prebuilt descriptors and events."""

    gpu: list[torch.Tensor]
    gpu_reference: list[torch.Tensor]
    host_backing: list[torch.Tensor]
    host: list[torch.Tensor]
    host_reference: list[torch.Tensor]
    ids: list[int]
    ids_gpu: torch.Tensor
    pointers_gpu: torch.Tensor
    batches: tuple[tuple[Any, ...], ...]
    native: Callable[..., None]
    producer: torch.cuda.Stream
    stream: torch.cuda.Stream
    ready: torch.cuda.Event
    start: torch.cuda.Event
    end: torch.cuda.Event
    direction: str
    reset_count: int = 0
    call_count: int = 0


if __name__ == "__main__":
    raise SystemExit(main())
