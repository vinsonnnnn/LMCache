# SPDX-License-Identifier: Apache-2.0
"""GPU data-mode checksum contracts with a mocked registration transport."""

# Standard
from unittest.mock import MagicMock
import hashlib

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform import kv_wrap


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kv_size", [1, 2])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gpu_data_checksums_follow_cleared_ranges_without_ipc(
    monkeypatch: pytest.MonkeyPatch,
    kv_size: int,
    dtype: str,
) -> None:
    """Hash GPU KV before/after clearing one chunk, without exporting handles.

    Args:
        monkeypatch: Fixture replacing the registration transport and IPC export.
        kv_size: One-plane MLA or two-plane classical KV layout.
        dtype: GPU element dtype, including BF16 without a NumPy dtype.

    Returns:
        None.

    Raises:
        AssertionError: If GPU data-mode requires IPC or hashes the wrong range.
    """
    transport = MagicMock(spec=RequestClient)
    chunk_size: MessagingFuture[int] = MessagingFuture()
    chunk_size.set_result(4)
    registered: MessagingFuture[bool] = MessagingFuture()
    registered.set_result(True)
    transport.get_chunk_size.return_value = chunk_size
    transport.register_kv_cache_engine_driven_context.return_value = registered
    transport.unregister_kv_cache_engine_driven_context.return_value = registered
    monkeypatch.setattr(
        RequestClientFactory, "create", MagicMock(return_value=transport)
    )
    export = MagicMock(side_effect=AssertionError("GPU data mode must not export IPC"))
    monkeypatch.setattr(kv_wrap, "wrap_one_kv_cache", export)
    client = ServerBenchClient(
        BenchConfig(
            rpc_url="tcp://127.0.0.1:1",
            http_url="",
            mode="gpu",
            transfer_mode="engine_driven",
            tp_size=1,
            use_mla=kv_size == 1,
            num_tokens=8,
            kvcache_shape_spec=f"({kv_size},8,2,2,4):{dtype}:2",
            num_blocks=8,
            block_size=2,
        ),
        lambda message: None,
    )
    try:
        client.start()
        request = client.create_request(0, "checksum-contract", "test")
        assert request is not None
        before = client.compute_checksums(request, 0, 8)
        assert before is not None and len(before) == 2
        client.zero_destination(request, 0, 4)
        after = client.compute_checksums(request, 0, 8)
        # 2 layers x 4 tokens x 2 heads x head_dim 4 x dtype bytes x K/V.
        zero_digest = hashlib.md5(bytes(2 * 4 * 2 * 4 * 2 * kv_size)).hexdigest()
        assert before[0] != zero_digest
        assert after == [zero_digest, before[1]]
        export.assert_not_called()
        transport.register_kv_cache.assert_not_called()
    finally:
        client.close()
