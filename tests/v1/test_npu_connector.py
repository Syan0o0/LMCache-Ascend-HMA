# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from unittest.mock import patch
import random

# Third Party
from lmcache.v1.memory_management import MemoryFormat, PinMemoryAllocator

# TODO (gingfung): once we have sglang kernel, re-enable test_sglang_connector_with_gpu_and_mla
from lmcache_tests.v1.test_gpu_connector import (
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_with_gpu_and_mla as original_test_vllm_paged_connector_v2_with_gpu_and_mla,
)
import pytest
import torch

# First Party
from lmcache_ascend.v1.npu_connector import (
    KVCacheFormat,
    _NPUV3GroupContext,
    SGLangLayerwiseNPUConnector,
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
    VLLMPagedMemNPUConnectorV3,
)
from tests.v1.utils import check_sglang_npu_kv_cache_equal, generate_sglang_npu_kv_cache


class _FakeMemoryObj:
    def __init__(self, tensors):
        self._tensors = tensors

    def get_tensor(self, index: int):
        return self._tensors[index]


@pytest.mark.parametrize("use_npu", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
def test_vllm_paged_connector_v2_with_npu_and_mla(use_npu, use_mla):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_with_gpu_and_mla(use_npu, use_mla)


@pytest.mark.parametrize("use_npu", [True])
def test_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)


def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)


@pytest.mark.parametrize("use_gpu", [True])
@pytest.mark.parametrize("use_mla", [True, False])
def test_sglang_layerwise_connector_with_npu(use_gpu, use_mla):
    """
    Test SGLang NPU integration with LMCache-Ascend.

    This test verifies the complete workflow of SGLang NPU with LMCache-Ascend:
    1. Generate SGLang NPU Layer-Concatenated format KV cache
    2. Test KV cache transfer from NPU to CPU (store)
    3. Test KV cache transfer from CPU to NPU (load)
    4. Verify the data integrity after round-trip transfer

    KV cache format: [2, layer_nums, num_blocks, block_size, num_heads, head_dim]
    """
    num_blocks = 100
    block_size = 16
    num_layers = 32
    num_heads = 8
    head_size = 128
    device = "npu"
    dtype = torch.bfloat16
    hidden_dim = num_heads * head_size

    num_tokens = num_blocks * block_size // 2
    chunk_size = 256

    allocator = PinMemoryAllocator(1024 * 1024 * 1024)

    gpu_kv_src = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )
    gpu_kv_dst = generate_sglang_npu_kv_cache(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
        device=device,
        dtype=dtype,
    )

    slot_mapping = random.sample(range(0, num_blocks * block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping, device=device, dtype=torch.int64)

    # Check the gpu_kv_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_sglang_npu_kv_cache_equal(
            gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
        )

    connector = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    connector2 = SGLangLayerwiseNPUConnector(
        hidden_dim,
        num_layers,
        use_gpu=use_gpu,
        chunk_size=chunk_size,
        dtype=dtype,
        device=device,
        use_mla=use_mla,
    )
    assert connector.use_mla == use_mla
    assert connector2.use_mla == use_mla

    # from gpu to cpu
    starts = []
    ends = []
    memory_objs = []

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        shape_single_layer = connector.get_shape(end - start)
        memory_objs_multi_layer = []

        for layer_id in range(num_layers):
            mem_obj_single_layer = allocator.allocate(
                shape_single_layer, dtype, fmt=MemoryFormat.KV_T2D
            )
            memory_objs_multi_layer.append(mem_obj_single_layer)

        starts.append(start)
        ends.append(end)
        memory_objs.append(memory_objs_multi_layer)

    memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]

    mem_obj_generator = connector.batched_from_gpu(
        memory_objs,
        starts,
        ends,
        kvcaches=gpu_kv_src,
        slot_mapping=slot_mapping,
        sync=True,
    )

    for layer_id in range(num_layers + 1):
        next(mem_obj_generator)

    # from cpu to gpu
    mem_obj_consumer = connector2.batched_to_gpu(
        starts,
        ends,
        kvcaches=gpu_kv_dst,
        slot_mapping=slot_mapping,
        sync=True,
    )

    next(mem_obj_consumer)
    for layer_id in range(num_layers):
        mem_obj_consumer.send(memory_objs[layer_id])

    # free all mem objs
    for mem_obj_multi_layer in memory_objs:
        for mem_obj in mem_obj_multi_layer:
            mem_obj.ref_count_down()

    assert allocator.memcheck()

    assert connector.gpu_buffer_allocator.memcheck()

    check_sglang_npu_kv_cache_equal(
        gpu_kv_src, gpu_kv_dst, slot_mapping, num_heads, head_size
    )

    allocator.close()


def test_vllm_paged_connector_v3_gdn_copy_helpers():
    connector = object.__new__(VLLMPagedMemNPUConnectorV3)

    conv_layer0 = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=torch.float16
    )
    ssm_layer0 = torch.tensor(
        [[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]], dtype=torch.float32
    )
    conv_layer1 = torch.tensor(
        [[12.0, 13.0], [22.0, 23.0], [32.0, 33.0]], dtype=torch.float16
    )
    ssm_layer1 = torch.tensor(
        [[102.0, 103.0], [202.0, 203.0], [302.0, 303.0]], dtype=torch.float32
    )
    connector.kvcaches = [
        [conv_layer0, ssm_layer0],
        [conv_layer1, ssm_layer1],
    ]

    attention_placeholder = torch.zeros((1,), dtype=torch.float16)
    conv_snapshot = torch.zeros((2, 2), dtype=torch.float16)
    ssm_snapshot = torch.zeros((2, 2), dtype=torch.float32)
    memory_obj = _FakeMemoryObj([attention_placeholder, conv_snapshot, ssm_snapshot])

    group_ctx = _NPUV3GroupContext(
        group_idx=1,
        layer_indices=[0, 1],
        num_layers=2,
        kv_format=KVCacheFormat.GDN_ALIGN_STATE,
        group_kind="gdn",
        num_tensors=2,
        memory_tensor_start=1,
        memory_tensor_end=3,
        block_size=2,
        tensor_names=["conv_state", "ssm_state"],
    )

    connector._copy_gdn_group_from_gpu(
        memory_obj,
        group_ctx,
        end=4,
        block_ids_by_group=([7, 8], [0, 1, 2]),
    )

    assert torch.equal(conv_snapshot[0], conv_layer0[1])
    assert torch.equal(conv_snapshot[1], conv_layer1[1])
    assert torch.equal(ssm_snapshot[0], ssm_layer0[1])
    assert torch.equal(ssm_snapshot[1], ssm_layer1[1])

    conv_layer0.zero_()
    conv_layer1.zero_()
    ssm_layer0.zero_()
    ssm_layer1.zero_()

    connector._copy_gdn_group_to_gpu(
        memory_obj,
        group_ctx,
        end=4,
        block_ids_by_group=([7, 8], [0, 1, 2]),
    )

    assert torch.equal(conv_layer0[1], conv_snapshot[0])
    assert torch.equal(conv_layer1[1], conv_snapshot[1])
    assert torch.equal(ssm_layer0[1], ssm_snapshot[0])
    assert torch.equal(ssm_layer1[1], ssm_snapshot[1])


def test_vllm_paged_connector_v3_attention_tuple_python_copy_helpers():
    connector = object.__new__(VLLMPagedMemNPUConnectorV3)

    class _DummyGroup:
        hidden_dim_size = 2

    class _DummyGroupsManager:
        kv_layer_groups = [_DummyGroup()]

    class _DummyMetadata:
        kv_layer_groups_manager = _DummyGroupsManager()

    connector.metadata = _DummyMetadata()
    connector.device = torch.device("cpu")
    connector.kvcaches = [
        (
            torch.tensor(
                [
                    [[[1.0, 2.0]], [[3.0, 4.0]]],
                    [[[5.0, 6.0]], [[7.0, 8.0]]],
                ],
                dtype=torch.float16,
            ),
            torch.tensor(
                [
                    [[[11.0, 12.0]], [[13.0, 14.0]]],
                    [[[15.0, 16.0]], [[17.0, 18.0]]],
                ],
                dtype=torch.float16,
            ),
        ),
        (
            torch.tensor(
                [
                    [[[21.0, 22.0]], [[23.0, 24.0]]],
                    [[[25.0, 26.0]], [[27.0, 28.0]]],
                ],
                dtype=torch.float16,
            ),
            torch.tensor(
                [
                    [[[31.0, 32.0]], [[33.0, 34.0]]],
                    [[[35.0, 36.0]], [[37.0, 38.0]]],
                ],
                dtype=torch.float16,
            ),
        ),
    ]

    slot_mapping = torch.tensor([1, 3], dtype=torch.long)
    memory_tensor = torch.zeros((2, 2, 2, 2), dtype=torch.float16)
    memory_obj = _FakeMemoryObj([memory_tensor])

    group_ctx = _NPUV3GroupContext(
        group_idx=0,
        layer_indices=[0, 1],
        num_layers=2,
        kv_format=KVCacheFormat.SEPARATE_KV,
        memory_tensor_start=0,
        memory_tensor_end=1,
    )

    connector._copy_attention_group_from_gpu_python(memory_obj, group_ctx, slot_mapping)

    assert torch.equal(memory_tensor[0, 0, 0], torch.tensor([3.0, 4.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[0, 0, 1], torch.tensor([7.0, 8.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[1, 1, 0], torch.tensor([33.0, 34.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[1, 1, 1], torch.tensor([37.0, 38.0], dtype=torch.float16))

    connector.kvcaches[0][0].zero_()
    connector.kvcaches[0][1].zero_()
    connector.kvcaches[1][0].zero_()
    connector.kvcaches[1][1].zero_()

    connector._copy_attention_group_to_gpu_python(memory_obj, group_ctx, slot_mapping)

    assert torch.equal(
        connector.kvcaches[0][0].reshape(-1, 2)[slot_mapping],
        memory_tensor[0, 0],
    )
    assert torch.equal(
        connector.kvcaches[1][1].reshape(-1, 2)[slot_mapping],
        memory_tensor[1, 1],
    )


def test_vllm_paged_connector_v3_attention_merged_python_copy_helpers():
    connector = object.__new__(VLLMPagedMemNPUConnectorV3)

    class _DummyGroup:
        hidden_dim_size = 2

    class _DummyGroupsManager:
        kv_layer_groups = [_DummyGroup()]

    class _DummyMetadata:
        kv_layer_groups_manager = _DummyGroupsManager()

    connector.metadata = _DummyMetadata()
    connector.device = torch.device("cpu")
    connector.kvcaches = [
        torch.tensor(
            [
                [
                    [[[1.0, 2.0]], [[3.0, 4.0]]],
                    [[[5.0, 6.0]], [[7.0, 8.0]]],
                ],
                [
                    [[[11.0, 12.0]], [[13.0, 14.0]]],
                    [[[15.0, 16.0]], [[17.0, 18.0]]],
                ],
            ],
            dtype=torch.float16,
        )
    ]

    slot_mapping = torch.tensor([0, 2], dtype=torch.long)
    memory_tensor = torch.zeros((2, 1, 2, 2), dtype=torch.float16)
    memory_obj = _FakeMemoryObj([memory_tensor])

    group_ctx = _NPUV3GroupContext(
        group_idx=0,
        layer_indices=[0],
        num_layers=1,
        kv_format=KVCacheFormat.MERGED_KV,
        memory_tensor_start=0,
        memory_tensor_end=1,
    )

    connector._copy_attention_group_from_gpu_python(memory_obj, group_ctx, slot_mapping)

    assert torch.equal(memory_tensor[0, 0, 0], torch.tensor([1.0, 2.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[0, 0, 1], torch.tensor([5.0, 6.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[1, 0, 0], torch.tensor([11.0, 12.0], dtype=torch.float16))
    assert torch.equal(memory_tensor[1, 0, 1], torch.tensor([15.0, 16.0], dtype=torch.float16))

    connector.kvcaches[0].zero_()
    connector._copy_attention_group_to_gpu_python(memory_obj, group_ctx, slot_mapping)

    merged_k = connector.kvcaches[0][0].reshape(-1, 2)
    merged_v = connector.kvcaches[0][1].reshape(-1, 2)
    assert torch.equal(merged_k[slot_mapping], memory_tensor[0, 0])
    assert torch.equal(merged_v[slot_mapping], memory_tensor[1, 0])
