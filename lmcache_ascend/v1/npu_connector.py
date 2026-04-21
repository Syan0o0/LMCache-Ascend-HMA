# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from enum import Enum, auto
import hashlib
import os
from typing import Any, List, Optional, Set, Tuple, Union

# Third Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector import (
    SGLangGPUConnector,
    SGLangLayerwiseGPUConnector,
    VLLMBufferLayerwiseGPUConnector,
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
    GPUConnectorInterface
)
from lmcache.v1.memory_management import GPUMemoryAllocator, MemoryFormat, MemoryObj
import torch

# First Party
from lmcache_ascend.v1.proxy_memory_obj import ProxyMemoryObj
from lmcache_ascend.v1.transfer_context import AscendBaseTransferContext
import lmcache_ascend.c_ops as lmc_ops

logger = init_logger(__name__)

_IS_310P = None
_ENABLE_TENSOR_SAMPLE_LOG = os.getenv("LMCACHE_TENSOR_SAMPLE_LOG", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_GDN_ALIGN_LOAD_LAST_ONLY = os.getenv(
    "LMCACHE_GDN_ALIGN_LOAD_LAST_ONLY", "1"
).lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _summarize_int_values(values: list[int], limit: int = 6) -> str:
    if not values:
        return "len=0 values=[]"
    if len(values) <= limit * 2:
        return f"len={len(values)} values={values}"
    return (
        f"len={len(values)} head={values[:limit]} tail={values[-limit:]}"
    )


def _build_edge_indices(length: int, count: int = 2) -> list[int]:
    if length <= 0:
        return []
    head = list(range(min(count, length)))
    tail_start = max(len(head), length - count)
    tail = list(range(tail_start, length))
    return sorted(set(head + tail))


def _tensor_sample_summary(
    tensor: torch.Tensor,
    value_limit: int = 6,
) -> str:
    detached = tensor.detach()
    original_shape = tuple(detached.shape)
    original_dtype = str(detached.dtype)
    original_device = str(detached.device)

    flat = detached.reshape(-1)
    sample_indices = _build_edge_indices(flat.numel(), count=value_limit)
    if sample_indices:
        sample_index_tensor = torch.tensor(
            sample_indices,
            dtype=torch.long,
            device=flat.device,
        )
        sample_cpu = flat.index_select(0, sample_index_tensor).to("cpu")
    else:
        sample_cpu = flat.to("cpu")

    if torch.is_floating_point(sample_cpu):
        sample_cpu = sample_cpu.to(torch.float32)
        has_nan = bool(torch.isnan(sample_cpu).any().item())
        has_inf = bool(torch.isinf(sample_cpu).any().item())
        finite_values = sample_cpu[torch.isfinite(sample_cpu)]
        absmax = (
            f"{float(finite_values.abs().max().item()):.6g}"
            if finite_values.numel() > 0
            else "nan"
        )
        display_values = [f"{float(v):.6g}" for v in sample_cpu.tolist()]
        hash_payload = sample_cpu.numpy().tobytes()
    else:
        has_nan = False
        has_inf = False
        sample_cpu = sample_cpu.to(torch.int64)
        absmax = (
            str(int(sample_cpu.abs().max().item()))
            if sample_cpu.numel() > 0
            else "0"
        )
        display_values = [str(int(v)) for v in sample_cpu.tolist()]
        hash_payload = sample_cpu.numpy().tobytes()

    digest = hashlib.blake2s(hash_payload, digest_size=6).hexdigest()
    return (
        f"shape={original_shape} dtype={original_dtype} device={original_device} "
        f"fp={digest} absmax={absmax} has_nan={has_nan} has_inf={has_inf} "
        f"sample_indices={sample_indices} values={display_values}"
    )


def _should_log_layer_position(layer_pos: int, total_layers: int) -> bool:
    return layer_pos == 0 or layer_pos == total_layers - 1


def _should_log_tensor_samples(layer_pos: int, total_layers: int) -> bool:
    return _ENABLE_TENSOR_SAMPLE_LOG and _should_log_layer_position(
        layer_pos, total_layers
    )


def _cpu_select_rows(tensor: torch.Tensor, rows: list[int]) -> torch.Tensor:
    cpu_tensor = tensor.detach().to("cpu")
    if not rows:
        return cpu_tensor.reshape(0, *cpu_tensor.shape[1:])
    row_index = torch.tensor(rows, dtype=torch.long)
    return cpu_tensor.index_select(0, row_index)


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        # First Party
        from lmcache_ascend import _build_info

        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P

def _get_slot_mappings_by_group_from_kwargs(
    kwargs,
) -> Tuple[torch.Tensor, ...]:
    slot_mappings_by_group = kwargs.get("slot_mappings_by_group")
    legacy_slot_mapping = kwargs.get("slot_mapping")

    if slot_mappings_by_group is None and legacy_slot_mapping is None:
        raise ValueError(
            "Either 'slot_mappings_by_group' or 'slot_mapping' should be provided in kwargs."
        )

    if slot_mappings_by_group is not None:
        if not isinstance(slot_mappings_by_group, tuple):
            raise ValueError(
                "'slot_mappings_by_group' should be a tuple of slot mappings."
            )
        normalized = tuple(
            _tensorize_slot_mapping(slot_mapping)
            for slot_mapping in slot_mappings_by_group
        )
    else:
        assert legacy_slot_mapping is not None
        normalized = (_tensorize_slot_mapping(legacy_slot_mapping),)

    if legacy_slot_mapping is not None:
        legacy_tensor = _tensorize_slot_mapping(legacy_slot_mapping)
        if len(normalized) != 1:
            raise ValueError(
                "Both 'slot_mapping' and multi-group "
                "'slot_mappings_by_group' were provided."
            )
        if not torch.equal(normalized[0], legacy_tensor):
            raise ValueError(
                "'slot_mapping' and 'slot_mappings_by_group[0]' do not match."
            )

    return normalized


def _get_block_ids_by_group_from_kwargs(
    kwargs,
    caller_name: str,
) -> Tuple[List[int], ...]:
    block_ids_by_group = kwargs.get("block_ids_by_group")
    legacy_block_ids = kwargs.get("block_ids")

    if block_ids_by_group is None and legacy_block_ids is None:
        raise ValueError(
            f"{caller_name} requires 'block_ids_by_group' for GDN state transfer."
        )

    if block_ids_by_group is not None:
        if not isinstance(block_ids_by_group, tuple):
            raise ValueError("'block_ids_by_group' should be a tuple of block id lists.")
        normalized = tuple(list(group_block_ids) for group_block_ids in block_ids_by_group)
    else:
        assert legacy_block_ids is not None
        normalized = (list(legacy_block_ids),)

    if legacy_block_ids is not None:
        if len(normalized) != 1:
            raise ValueError(
                "Both 'block_ids' and multi-group 'block_ids_by_group' were provided."
            )
        if normalized[0] != list(legacy_block_ids):
            raise ValueError(
                "'block_ids' and 'block_ids_by_group[0]' do not match."
            )

    return normalized


def _get_legacy_single_slot_mapping_from_kwargs(
    kwargs,
    caller_name: str,
) -> torch.Tensor:
    slot_mappings_by_group = _get_slot_mappings_by_group_from_kwargs(kwargs)
    if len(slot_mappings_by_group) != 1:
        raise NotImplementedError(
            f"{caller_name} does not support multi-group KV cache yet. "
            "Please use NPU connector V3."
        )
    return slot_mappings_by_group[0]


def _assert_single_group_or_raise(
    kwargs,
    caller_name: str,
) -> None:
    slot_mappings_by_group = _get_slot_mappings_by_group_from_kwargs(kwargs)
    if len(slot_mappings_by_group) != 1:
        raise NotImplementedError(
            f"{caller_name} does not support multi-group KV cache yet. "
            "Please use NPU connector V3."
        )


def _tensorize_slot_mapping(
    slot_mapping: Union[torch.Tensor, List[int]],
) -> torch.Tensor:
    if isinstance(slot_mapping, torch.Tensor):
        return slot_mapping.to(dtype=torch.long)
    return torch.tensor(slot_mapping, dtype=torch.long)

class KVCacheFormat(Enum):
    """
    The storage format enumeration of KV cache is used to distinguish
    the KV cache data structures of different versions of vLLM.

    The order of enum values MUST match the KVCacheFormat
    definition in kernels/types.h to ensure correct interoperability
    between Python and C++ code.
    """

    UNDEFINED = 0

    MERGED_KV = auto()
    """merge format (eg: vLLM 0.9.2 ...)
    layer: [num_kv, num_blocks, block_size, num_heads, head_dim]
    """

    SEPARATE_KV = auto()
    """Separation format (eg: vLLM 0.11.0+ ...)
    layer: tuple: (K_tensor, V_tensor)
    - K_tensor.shape = [num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [num_blocks, block_size, num_heads, head_dim]

    eg: kvcaches[0] = (K, V)

    SGLang NPU Layer-Concatenated
    kvcaches = [K_all_layers, V_all_layers]
    - K_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    - V_tensor.shape = [layer_nums, num_blocks, block_size, num_heads, head_dim]
    """

    GDN_ALIGN_STATE = auto()
    """Gated DeltaNet align-state format.

    layer: sequence of state tensors, usually [conv_state, ssm_state]
    - conv_state.shape = [num_blocks, ...]
    - ssm_state.shape = [num_blocks, ...]
    """

    def is_separate_format(self) -> bool:
        return self == KVCacheFormat.SEPARATE_KV

    def is_merged_format(self) -> bool:
        return self == KVCacheFormat.MERGED_KV

    def is_gdn_state_format(self) -> bool:
        return self == KVCacheFormat.GDN_ALIGN_STATE

    @staticmethod
    def detect(
        kvcaches: List[
            Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
        ],
        use_mla: bool = False,
        group_kind: Optional[str] = None,
    ) -> "KVCacheFormat":
        if not kvcaches:
            return KVCacheFormat.UNDEFINED

        first_cache = kvcaches[0]

        if group_kind == "gdn":
            if isinstance(first_cache, (tuple, list)):
                if len(first_cache) >= 1 and all(
                    isinstance(tensor, torch.Tensor) for tensor in first_cache
                ):
                    return KVCacheFormat.GDN_ALIGN_STATE
            return KVCacheFormat.UNDEFINED

        # SGLang NPU: kvcaches = [K_tensor, V_tensor]
        if isinstance(kvcaches, list) and len(kvcaches) == 2:
            if isinstance(first_cache, torch.Tensor) and first_cache.ndim == 5:
                return KVCacheFormat.SEPARATE_KV

        if isinstance(first_cache, tuple):
            return KVCacheFormat.SEPARATE_KV
        elif isinstance(first_cache, torch.Tensor):
            ndim = first_cache.ndim
            shape = first_cache.shape

            # MLA detect
            # MLA Shape: [num_blocks, block_size, head_size] (3D)
            #         or: [1, num_blocks, block_size, head_size] (4D with first dim = 1)
            is_mla_shape = (ndim == 3) or (ndim == 4 and shape[0] == 1)
            if use_mla or is_mla_shape:
                return KVCacheFormat.MERGED_KV

            # Flash Attention：[2, num_blocks, block_size, num_heads, head_size]
            if ndim == 5 and shape[0] == 2:
                return KVCacheFormat.MERGED_KV

            # Flash Infer：[num_blocks, 2, block_size, num_heads, head_size]
            if ndim == 5 and shape[1] == 2:
                return KVCacheFormat.MERGED_KV

        return KVCacheFormat.UNDEFINED


@dataclass
class _NPUV3GroupContext:
    group_idx: int
    layer_indices: List[int]
    num_layers: int
    kv_format: KVCacheFormat
    group_kind: str = "attention"
    num_tensors: int = 1
    memory_tensor_start: int = 0
    memory_tensor_end: int = 1
    block_size: int = 0
    tensor_names: Optional[List[str]] = None
    kv_cache_pointers_on_device: Optional[torch.Tensor] = None
    page_buffer_size: int = 0
    tmp_buffer: Optional[torch.Tensor] = None


class VLLMBufferLayerwiseNPUConnector(VLLMBufferLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_dim_size, num_layers, use_gpu, use_double_buffer, **kwargs
        )
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED
        self.use_mla = bool(kwargs.get("use_mla", False))
        self.fused_rotary_emb: Any = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            self.kv_format = KVCacheFormat.detect(kv_caches)
            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError("Could not detect KV cache format.")

            ref_tensor = (
                kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]
            )
            self.kv_device = ref_tensor.device

            first_layer_cache = kv_caches[0]

            # flash attention: [num_layers, 2, num_blocks,
            # block_size, num_heads, head_size]
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                k_cache_shape_per_layer = first_layer_cache[0].shape
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            self.vllm_two_major = True

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def _prepare_transfer_context(self, kwargs) -> torch.Tensor:
        """
        Initialize context for KV cache transfer, validate required
        parameters and lazy init buffer.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        if self.kvcaches is None:
            raise ValueError("kvcaches should be provided in kwargs or initialized.")

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")
        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "_prepare_transfer_context"
        )

        self._lazy_initialize_buffer(self.kvcaches)
        # return kwargs["slot_mapping"]
        return slot_mapping

    def _get_full_slot_mapping(
        self,
        slot_mapping: torch.Tensor,
        starts: List[int],
        ends: List[int],
        mode: str = "slice",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate full continuous slot mapping tensor and calculate total token count.
        Supports two modes for different transfer directions (to/from GPU).
        """
        if mode == "slice":
            slot_mapping_full = slot_mapping[starts[0] : ends[-1]]
        elif mode == "concat":
            slot_mapping_chunks = [
                slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
            ]
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)
        else:
            raise ValueError(
                f"Unsupported slot mapping mode: {mode}, only 'slice'/'concat' allowed"
            )

        num_tokens = len(slot_mapping_full)
        return slot_mapping_full, num_tokens

    def _allocate_gpu_buffers(
        self, num_tokens: int, count: int = 1
    ) -> Union[object, list[object]]:
        """
        Allocate specified number of GPU buffers for KV cache with shape
        calculated by token count. Performs strict assertion checks for
        valid buffer allocation.
        """
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None, (
            "GPU buffer allocator not initialized"
        )
        buffers = []
        for _ in range(count):
            buf_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_2TD
            )
            assert buf_obj is not None, "Failed to allocate GPU buffer in GPUConnector"
            assert buf_obj.tensor is not None, "GPU buffer object has no valid tensor"
            buffers.append(buf_obj)
        return buffers[0] if count == 1 else buffers

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping_full, num_all_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="slice"
        )

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]
        load_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj: Any = None
        compute_gpu_buffer_obj, load_gpu_buffer_obj = self._allocate_gpu_buffers(
            num_all_tokens, count=2
        )

        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kv_device
            )
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kv_device
            )

        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        slot_mapping = self._prepare_transfer_context(kwargs)

        buf_start = 0
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(start, end, device=self.kv_device, dtype=torch.int64)
                )

        slot_mapping_full, num_tokens = self._get_full_slot_mapping(
            slot_mapping, starts, ends, mode="concat"
        )

        tmp_gpu_buffer_obj = self._allocate_gpu_buffers(num_tokens, count=1)

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                )

                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield


class VLLMPagedMemNPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        if is_310p():
            assert "num_kv_head" in kwargs, ("num_kv_head should be provided in 310p",)
            assert "head_size" in kwargs, ("head_size should be provided in 310p",)
            self.num_kv_head = kwargs["num_kv_head"]
            self.head_size = kwargs["head_size"]
            self.dtype = kwargs["dtype"]
            self.device = kwargs["device"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            num_kv_head=num_kv_head,
            head_size=head_size,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError(
                "Undefined KV cache format detected. "
                "Unable to determine the format of input kv_caches."
            )

        if self.kv_format.is_separate_format():
            self.kvcaches_device = kv_caches[0][0].device
        else:
            self.kvcaches_device = kv_caches[0].device

        assert self.kvcaches_device.type == "npu", "The device should be Ascend NPU."
        idx = self.kvcaches_device.index

        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        if self.kv_format == KVCacheFormat.SEPARATE_KV:
            self.kv_size = 2
            pointers_list = []
            for k, v in kv_caches:
                pointers_list.append(k.data_ptr())
                pointers_list.append(v.data_ptr())

            self.kv_cache_pointers = torch.empty(
                self.num_layers * self.kv_size, dtype=torch.int64, device="cpu"
            )
        else:
            self.kv_size = 1
            pointers_list = [t.data_ptr() for t in kv_caches]

            self.kv_cache_pointers = torch.empty(
                self.num_layers, dtype=torch.int64, device="cpu"
            )

        self.kv_cache_pointers.numpy()[:] = pointers_list

        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.kv_cache_pointers.shape, dtype=torch.int64, device=self.kvcaches_device
        )

        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        first_tensor = (
            kv_caches[0][0] if self.kv_format.is_separate_format() else kv_caches[0]
        )

        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            # kv_caches[0].shape: [1, num_pages, page_size, head_size] (vllm-Ascend)
            self.page_buffer_size = kv_caches[0].shape[-3] * kv_caches[0].shape[-2]
        else:
            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                # kv_caches[0]: [tuple(k,v)]
                # 310P: [num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() >= 2
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[0] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[0] * first_tensor.shape[1]
                    )

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
                # 310P: [2, num_blocks, num_kv_heads * head_size // 16, block_size, 16]
                # 910B: [2, num_blocks, block_size, num_kv_heads, head_size]
                assert first_tensor.dim() == 5
                if is_310p():
                    self.block_size = first_tensor.shape[-2]
                    self.page_buffer_size = first_tensor.shape[1] * self.block_size
                else:
                    self.page_buffer_size = (
                        first_tensor.shape[1] * first_tensor.shape[2]
                    )

        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format "
                    "in order to be processed by VLLMPagedMemNPUConnector."
                )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemNPUConnectorV2.to_gpu_310p"
        )

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        tmp_gpu_buffer.copy_(memory_obj.tensor)

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu_310p(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemNPUConnectorV2.to_gpu_310p"
        )

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        assert self.gpu_buffer.device == self.kvcaches_device

        tmp_gpu_buffer = torch.empty(
            memory_obj.tensor.size(), dtype=self.dtype, device=self.device
        )

        lmc_ops.multi_layer_kv_transfer_310p(
            tmp_gpu_buffer,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            True,
            self.use_mla,
            self.num_kv_head,
            self.head_size,
            self.block_size,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

        memory_obj.tensor.copy_(tmp_gpu_buffer)
        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemNPUConnector."
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in "
                    " order to be processed by VLLMPagedMemNPUConnector."
                )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemNPUConnectorV2.to_gpu_310p"
        )

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.kvcaches_device,
            self.page_buffer_size,
            False,
            self.use_mla,
            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemNPUConnectorV2.from_gpu"
        )

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("KV cache format is not initialized!")

        with torch.cuda.stream(self.store_stream):
            # No staging buffer or token count mismatch
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                    self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                )
            else:
                assert self.gpu_buffer.device == self.kvcaches_device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.fused_multi_layer_kv_transfer(
                    memory_obj.tensor,  # dst: CPU buffer
                    tmp_gpu_buffer,  # staging cache
                    kv_cache_pointers,  # src: paged KV cache
                    slot_mapping[start:end],
                    self.kvcaches_device,
                    self.page_buffer_size,
                    True,  # from_gpu
                    self.use_mla,
                    self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                )

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        # Check if any memory objects are ProxyMemoryObjs (deferred P2P fetch)
        has_proxy = any(isinstance(m, ProxyMemoryObj) for m in memory_objs)

        if has_proxy:
            assert not is_310p(), "Batched P2P transfer is not supported on 310P."

            self._remote_batched_to_gpu(memory_objs, starts, ends, **kwargs)

            # NOTE (gingfung): Ensure the compute stream waits for
            # load_stream's KV scatter to complete before attention
            # reads the same pages.
            # load_stream.synchronize() in _remote_batched_to_gpu is
            # host-side only, the compute stream has no knowledge of it
            # and can race ahead.
            torch.npu.current_stream().wait_stream(self.load_stream)
        else:
            # _assert_single_group_or_raise(
            #     kwargs, "VLLMPagedMemNPUConnectorV2.batched_to_gpu"
            # )
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in zip(
                    memory_objs, starts, ends, strict=False
                ):
                    if is_310p():
                        self.to_gpu_310p(memory_obj, start, end, **kwargs)
                    else:
                        self.to_gpu(memory_obj, start, end, **kwargs)
            self.load_stream.synchronize()

    def _clear_proxy_batch(self, batch) -> None:
        """Clear the backing objects of the proxy batch."""
        for proxy, _, _ in batch:
            proxy.clear_backing_obj()
        return None

    def _scatter_proxy_batch(self, batch, event, **kwargs):
        """Wait for a read event, scatter proxies to KV cache.

        Enqueues work on ``load_stream``.  The caller is responsible for
        recording a scatter-done event afterwards if needed for
        cross-stream synchronization.
        """
        if event is not None:
            self.load_stream.wait_event(event)
        with torch.cuda.stream(self.load_stream):
            for proxy, start, end in batch:
                self.to_gpu(proxy.backing_obj, start, end, **kwargs)

    def _remote_batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """Handle batched_to_gpu when ProxyMemoryObjs are present.

        Uses a ping-pong pipeline with **event-based** cross-stream
        synchronization to overlap remote data fetching (on the HCCL
        transport stream) with KV cache scatter (on the load stream).


        Two pools of PIPELINE_DEPTH buffers are allocated from the
        transfer context's registered memory and alternated (ping-pong).
        This limits peak memory to 2 x PIPELINE_DEPTH chunks regardless
        of the total number of proxy objects.

        After all proxy objects are processed, sends the Done signal
        to release the remote peer's pinned resources.
        """
        transfer_contexts: Set[AscendBaseTransferContext] = set()

        # Separate proxy and non-proxy items
        proxy_items = []
        non_proxy_items = []
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if isinstance(memory_obj, ProxyMemoryObj):
                transfer_contexts.add(memory_obj.transfer_context)
                proxy_items.append((memory_obj, start, end))
            else:
                non_proxy_items.append((memory_obj, start, end))

        if proxy_items:
            # Get the transfer context for buffer allocation
            first_ctx = proxy_items[0][0].transfer_context

            # Derive pipeline depth from NPU buffer capacity so that
            # two full ping-pong pools fit in registered memory.
            pipeline_depth = first_ctx.max_pipeline_depth
            logger.debug(
                "P2P pipeline depth = %d (proxy_items=%d)",
                pipeline_depth,
                len(proxy_items),
            )

            # Allocate ping-pong buffer pools.
            # Initialized to None so the finally block can safely skip
            # release if allocation itself fails.
            pool_size = min(pipeline_depth, len(proxy_items))
            pool_a = None
            pool_b = None

            try:
                pool_a = first_ctx.allocate_buffers(pool_size)
                pool_b = first_ctx.allocate_buffers(pool_size)

                pools = [pool_a, pool_b]
                current_pool = 0

                # Group proxy items into micro-batches
                micro_batches = [
                    proxy_items[i : i + pipeline_depth]
                    for i in range(0, len(proxy_items), pipeline_depth)
                ]

                prev_read_event = None
                prev_batch = None

                # Per-pool scatter-done events: prevent the next RDMA
                # write into a pool from racing with a scatter that is
                # still reading from the same pool on load_stream.
                # Events are pre-allocated and re-recorded each iteration.
                channel = proxy_items[0][0]._transfer_channel
                transport_stream = getattr(channel, "transport_stream", None)
                pool_scatter_events = [
                    torch.npu.Event(),
                    torch.npu.Event(),
                ]
                pool_scatter_recorded = [False, False]

                for batch_idx, batch in enumerate(micro_batches):
                    pool = pools[current_pool]

                    # Ensure the previous scatter from this pool has
                    # finished before RDMA overwrites the pool buffers.
                    if (
                        pool_scatter_recorded[current_pool]
                        and transport_stream is not None
                    ):
                        transport_stream.wait_event(pool_scatter_events[current_pool])

                    # Assign backing buffers from current pool to proxies
                    for i, (proxy, _, _) in enumerate(batch):
                        proxy.set_backing_obj(pool[i])

                    proxies = [item[0] for item in batch]

                    # Submit RDMA read for current batch → transport_stream.
                    cur_read_event = ProxyMemoryObj.submit_resolve_batch(proxies)

                    # While the current batch is being read on
                    # transport_stream, scatter the previous batch on
                    # load_stream (waits for its RDMA read event).
                    if prev_batch is not None:
                        self._scatter_proxy_batch(
                            prev_batch,
                            prev_read_event,
                            **kwargs,
                        )
                        pool_scatter_events[1 - current_pool].record(self.load_stream)
                        pool_scatter_recorded[1 - current_pool] = True
                        self._clear_proxy_batch(prev_batch)

                    prev_read_event = cur_read_event
                    prev_batch = batch
                    current_pool = 1 - current_pool  # toggle ping-pong

                # Drain: scatter the last micro-batch.
                if prev_batch is not None:
                    self._scatter_proxy_batch(
                        prev_batch,
                        prev_read_event,
                        **kwargs,
                    )
                    self._clear_proxy_batch(prev_batch)
            finally:
                # Guarantee ping-pong buffers are returned and the Done
                # signal is sent even if the pipeline raises or
                # allocate_buffers itself fails.  Without this, an
                # exception would leak NPU pages and leave the sender's
                # pinned resources stuck until its TTL expires.
                self.load_stream.synchronize()
                if pool_a is not None:
                    first_ctx.release_buffers(pool_a)
                if pool_b is not None:
                    first_ctx.release_buffers(pool_b)

                for proxy, _, _ in proxy_items:
                    proxy.mark_consumed()

                for ctx in transfer_contexts:
                    ctx.send_done_now()

        # Process non-proxy items on load_stream (no pipelining needed)
        if non_proxy_items:
            with torch.cuda.stream(self.load_stream):
                for memory_obj, start, end in non_proxy_items:
                    self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        _assert_single_group_or_raise(
            kwargs, "VLLMPagedMemNPUConnectorV2.batched_from_gpu"
        )
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            if is_310p():
                self.from_gpu_310p(memory_obj, start, end, **kwargs)
            else:
                self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemNPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: LMCacheEngineMetadata,
        device: torch.device,
        use_gpu: bool = False,
    ):
        assert device.type == "npu", "The device should be Ascend NPU."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.kvcaches: Optional[
            List[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]]
        ] = None
        self.group_contexts: Optional[List[_NPUV3GroupContext]] = None
        self.init = False
        self._gdn_op_unavailable_warned = False

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemNPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu)

    def _ensure_group_slot_mappings(
        self,
        kwargs,
        caller_name: str,
    ) -> Tuple[torch.Tensor, ...]:
        slot_mappings_by_group = _get_slot_mappings_by_group_from_kwargs(kwargs)
        if self.metadata.get_num_groups() != len(slot_mappings_by_group):
            raise ValueError(
                f"{caller_name} received {len(slot_mappings_by_group)} slot mapping "
                f"groups, but metadata expects {self.metadata.get_num_groups()}."
            )
        return slot_mappings_by_group

    def _describe_runtime_kv_cache(self, kv_cache) -> str:
        if isinstance(kv_cache, torch.Tensor):
            return (
                "Tensor("
                f"shape={tuple(kv_cache.shape)}, "
                f"dtype={kv_cache.dtype}, "
                f"device={kv_cache.device})"
            )

        if isinstance(kv_cache, (tuple, list)):
            tensor_descriptions: List[str] = []
            for idx, tensor in enumerate(kv_cache):
                if isinstance(tensor, torch.Tensor):
                    tensor_descriptions.append(
                        f"{idx}:{tuple(tensor.shape)}/{tensor.dtype}/{tensor.device}"
                    )
                else:
                    tensor_descriptions.append(f"{idx}:{type(tensor)}")
            return f"{type(kv_cache).__name__}[" + ", ".join(tensor_descriptions) + "]"

        return str(type(kv_cache))

    def _build_group_pointer_tensor(
        self,
        group_kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]],
        kv_format: KVCacheFormat,
    ) -> torch.Tensor:
        if kv_format == KVCacheFormat.SEPARATE_KV:
            pointers_list: List[int] = []
            for k_tensor, v_tensor in group_kvcaches:
                pointers_list.append(k_tensor.data_ptr())
                pointers_list.append(v_tensor.data_ptr())
            kv_cache_pointers = torch.empty(
                len(group_kvcaches) * 2,
                dtype=torch.int64,
                device="cpu",
            )
        else:
            pointers_list = [tensor.data_ptr() for tensor in group_kvcaches]
            kv_cache_pointers = torch.empty(
                len(group_kvcaches),
                dtype=torch.int64,
                device="cpu",
            )

        kv_cache_pointers.numpy()[:] = pointers_list
        kv_cache_pointers_on_device = torch.empty(
            kv_cache_pointers.shape,
            dtype=torch.int64,
            device=self.device,
        )
        kv_cache_pointers_on_device.copy_(kv_cache_pointers)
        return kv_cache_pointers_on_device

    def _extract_group_transfer_params(
        self,
        group_kvcaches: List[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]],
        kv_format: KVCacheFormat,
    ) -> int:
        if is_310p():
            raise NotImplementedError(
                "VLLMPagedMemNPUConnectorV3 first version does not support 310P."
            )

        if kv_format == KVCacheFormat.GDN_ALIGN_STATE:
            return 0

        if self.use_mla:
            representative = group_kvcaches[0]
            assert isinstance(representative, torch.Tensor)
            page_buffer_size = representative.shape[-3] * representative.shape[-2]
            return page_buffer_size

        if kv_format == KVCacheFormat.SEPARATE_KV:
            representative = group_kvcaches[0][0]
            page_buffer_size = representative.shape[0] * representative.shape[1]
            return page_buffer_size

        representative = group_kvcaches[0]
        assert isinstance(representative, torch.Tensor)
        if representative.shape[0] == 2:
            page_buffer_size = representative.shape[1] * representative.shape[2]
            return page_buffer_size

        page_buffer_size = representative.shape[0] * representative.shape[2]
        return page_buffer_size

    def _initialize_group_contexts(self):
        if self.init:
            return

        if is_310p():
            raise NotImplementedError(
                "VLLMPagedMemNPUConnectorV3 first version does not support 310P."
            )

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        assert self.metadata.kv_layer_groups_manager.kv_layer_groups, (
            "kv_layer_groups_manager must be built before using NPU connector V3."
        )

        group_transfer_shapes = self.metadata.get_group_transfer_shapes(self.chunk_size)
        group_transfer_dtypes = self.metadata.get_group_transfer_dtypes()
        group_transfer_index_ranges = self.metadata.get_group_transfer_index_ranges()
        if self.metadata.kv_group_block_sizes:
            block_sizes_by_group = self.metadata.kv_group_block_sizes
        else:
            block_sizes_by_group = tuple(0 for _ in range(self.metadata.get_num_groups()))

        self.group_contexts = []
        group_kinds = self.metadata.get_group_kinds()
        for group_idx, group in enumerate(
            self.metadata.kv_layer_groups_manager.kv_layer_groups
        ):
            group_kvcaches = [
                tensor
                for layer_idx, tensor in enumerate(self.kvcaches)
                if layer_idx in group.layer_indices
            ]
            group_kind = group_kinds[group_idx]
            logger.info(
                "NPU V3 initializing KV group %d/%d: kind=%s, block_size=%s, "
                "layer_names=%s, layer_indices=%s",
                group_idx,
                len(self.metadata.kv_layer_groups_manager.kv_layer_groups),
                group_kind,
                block_sizes_by_group[group_idx],
                group.layer_names,
                group.layer_indices,
            )
            logger.info(
                "NPU V3 group %d tensor_specs=%s",
                group_idx,
                [
                    (tensor_spec.name, tuple(tensor_spec.shape), str(tensor_spec.dtype))
                    for tensor_spec in group.tensor_specs
                ],
            )
            for local_idx, layer_cache in enumerate(group_kvcaches):
                logger.info(
                    "NPU V3 group %d runtime cache[%d]=%s",
                    group_idx,
                    local_idx,
                    self._describe_runtime_kv_cache(layer_cache),
                )
            kv_format = KVCacheFormat.detect(
                group_kvcaches,
                use_mla=self.use_mla,
                group_kind=group_kind,
            )
            if kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    f"Could not detect KV cache format for KV group {group_idx}."
                )
            kv_cache_pointers_on_device = None
            if kv_format != KVCacheFormat.GDN_ALIGN_STATE:
                kv_cache_pointers_on_device = self._build_group_pointer_tensor(
                    group_kvcaches, kv_format
                )
            page_buffer_size = self._extract_group_transfer_params(
                group_kvcaches, kv_format
            )
            tmp_buffer = None
            if self.use_gpu and kv_format != KVCacheFormat.GDN_ALIGN_STATE:
                tmp_buffer = torch.empty(
                    group_transfer_shapes[group_idx][0],
                    dtype=group_transfer_dtypes[group_idx][0],
                    device=self.device,
                )
            memory_tensor_start, memory_tensor_end = group_transfer_index_ranges[group_idx]
            logger.info(
                "NPU V3 group %d detected kv_format=%s, transfer_shapes=%s, "
                "transfer_dtypes=%s, memory_tensor_range=%s, page_buffer_size=%s",
                group_idx,
                kv_format.name,
                [tuple(shape) for shape in group_transfer_shapes[group_idx]],
                [str(dtype) for dtype in group_transfer_dtypes[group_idx]],
                (memory_tensor_start, memory_tensor_end),
                page_buffer_size,
            )

            self.group_contexts.append(
                _NPUV3GroupContext(
                    group_idx=group_idx,
                    layer_indices=list(group.layer_indices),
                    num_layers=group.num_layers,
                    kv_format=kv_format,
                    group_kind=group_kind,
                    num_tensors=group.num_tensors,
                    memory_tensor_start=memory_tensor_start,
                    memory_tensor_end=memory_tensor_end,
                    block_size=block_sizes_by_group[group_idx],
                    tensor_names=[
                        tensor_spec.name
                        for tensor_spec in group.tensor_specs
                    ],
                    kv_cache_pointers_on_device=kv_cache_pointers_on_device,
                    page_buffer_size=page_buffer_size,
                    tmp_buffer=tmp_buffer,
                )
            )

        self.init = True
        logger.info("Initialized NPU connector V3 group contexts successfully")

    def _get_group_memory_tensors(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
    ) -> List[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        for tensor_idx in range(group_ctx.memory_tensor_start, group_ctx.memory_tensor_end):
            tensor = memory_obj.get_tensor(tensor_idx)
            if tensor is None:
                raise ValueError(
                    f"Missing memory tensor {tensor_idx} for KV group {group_ctx.group_idx}."
                )
            tensors.append(tensor)
        return tensors

    def _get_attention_kv_views(
        self,
        layer_cache: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]],
        group_ctx: _NPUV3GroupContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_dim_size = self.metadata.kv_layer_groups_manager.kv_layer_groups[
            group_ctx.group_idx
        ].hidden_dim_size

        if group_ctx.kv_format == KVCacheFormat.SEPARATE_KV:
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                raise ValueError(
                    f"Attention group {group_ctx.group_idx} expects a (K, V) pair, "
                    f"got {type(layer_cache)}."
                )
            key_tensor = layer_cache[0]
            value_tensor = layer_cache[1]
            if not isinstance(key_tensor, torch.Tensor) or not isinstance(
                value_tensor, torch.Tensor
            ):
                raise ValueError(
                    f"Attention group {group_ctx.group_idx} expects tensor K/V cache."
                )
            return (
                key_tensor.reshape(-1, hidden_dim_size),
                value_tensor.reshape(-1, hidden_dim_size),
            )

        if not isinstance(layer_cache, torch.Tensor):
            raise ValueError(
                f"Attention group {group_ctx.group_idx} expects merged tensor cache, "
                f"got {type(layer_cache)}."
            )

        if layer_cache.ndim == 5 and layer_cache.shape[0] == 2:
            key_tensor = layer_cache[0]
            value_tensor = layer_cache[1]
        elif layer_cache.ndim == 5 and layer_cache.shape[1] == 2:
            key_tensor = layer_cache[:, 0]
            value_tensor = layer_cache[:, 1]
        else:
            raise NotImplementedError(
                "VLLMPagedMemNPUConnectorV3 Python attention transfer currently "
                "supports only standard merged/separate KV formats."
            )

        return (
            key_tensor.reshape(-1, hidden_dim_size),
            value_tensor.reshape(-1, hidden_dim_size),
        )

    def _get_attention_group_transfer_tensor(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
    ) -> torch.Tensor:
        memory_obj_tensors = self._get_group_memory_tensors(memory_obj, group_ctx)
        if len(memory_obj_tensors) != 1:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} expects exactly one transfer "
                f"tensor, got {len(memory_obj_tensors)}."
            )

        memory_obj_tensor = memory_obj_tensors[0]
        if not self.use_mla and memory_obj_tensor.shape[0] != 2:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} expects KV_2LTD tensor with "
                f"leading kv dim=2, got shape {tuple(memory_obj_tensor.shape)}."
            )
        return memory_obj_tensor

    def _run_attention_group_to_gpu_op(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
    ) -> None:
        if group_ctx.kv_cache_pointers_on_device is None:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} is missing KV cache pointers."
            )
        if group_ctx.page_buffer_size <= 0:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} has invalid page_buffer_size "
                f"{group_ctx.page_buffer_size}."
            )

        memory_obj_tensor = self._get_attention_group_transfer_tensor(
            memory_obj, group_ctx
        )
        slot_mapping = slot_mapping.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        expected_tokens = memory_obj_tensor.shape[2]
        if slot_mapping.numel() != expected_tokens:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} expects {expected_tokens} "
                f"slot mappings, got {slot_mapping.numel()}."
            )

        lmc_ops.multi_layer_kv_transfer(
            memory_obj_tensor,
            group_ctx.kv_cache_pointers_on_device,
            slot_mapping,
            self.device,
            group_ctx.page_buffer_size,
            False,
            self.use_mla,
            group_ctx.kv_format.value,
        )

    def _run_attention_group_from_gpu_op(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
    ) -> None:
        if group_ctx.kv_cache_pointers_on_device is None:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} is missing KV cache pointers."
            )
        if group_ctx.page_buffer_size <= 0:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} has invalid page_buffer_size "
                f"{group_ctx.page_buffer_size}."
            )

        memory_obj_tensor = self._get_attention_group_transfer_tensor(
            memory_obj, group_ctx
        )
        slot_mapping = slot_mapping.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        expected_tokens = memory_obj_tensor.shape[2]
        if slot_mapping.numel() != expected_tokens:
            raise ValueError(
                f"Attention group {group_ctx.group_idx} expects {expected_tokens} "
                f"slot mappings, got {slot_mapping.numel()}."
            )

        can_use_fused = (
            group_ctx.tmp_buffer is not None
            and not memory_obj_tensor.is_cuda
            and group_ctx.tmp_buffer.shape == memory_obj_tensor.shape
        )

        if can_use_fused:
            lmc_ops.fused_multi_layer_kv_transfer(
                memory_obj_tensor,
                group_ctx.tmp_buffer,
                group_ctx.kv_cache_pointers_on_device,
                slot_mapping,
                self.device,
                group_ctx.page_buffer_size,
                True,
                self.use_mla,
                group_ctx.kv_format.value,
            )
            return

        lmc_ops.multi_layer_kv_transfer(
            memory_obj_tensor,
            group_ctx.kv_cache_pointers_on_device,
            slot_mapping,
            self.device,
            group_ctx.page_buffer_size,
            True,
            self.use_mla,
            group_ctx.kv_format.value,
        )

    def _copy_attention_group_from_gpu(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
        req_id: Optional[str] = None,
    ) -> None:
        try:
            self._run_attention_group_from_gpu_op(
                memory_obj,
                group_ctx,
                slot_mapping,
            )
        except Exception as exc:
            logger.warning(
                "Attention group operator store failed, falling back to Python copy. "
                "req_id=%s group=%d error=%s",
                req_id,
                group_ctx.group_idx,
                exc,
            )
            self._copy_attention_group_from_gpu_python(
                memory_obj,
                group_ctx,
                slot_mapping,
                req_id=req_id,
            )

    def _copy_attention_group_to_gpu(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
        req_id: Optional[str] = None,
    ) -> None:
        try:
            self._run_attention_group_to_gpu_op(
                memory_obj,
                group_ctx,
                slot_mapping,
            )
        except Exception as exc:
            logger.warning(
                "Attention group operator load failed, falling back to Python copy. "
                "req_id=%s group=%d error=%s",
                req_id,
                group_ctx.group_idx,
                exc,
            )
            self._copy_attention_group_to_gpu_python(
                memory_obj,
                group_ctx,
                slot_mapping,
                req_id=req_id,
            )

    def _copy_attention_group_from_gpu_python(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
        req_id: Optional[str] = None,
    ) -> None:
        assert self.kvcaches is not None
        memory_obj_tensor = self._get_attention_group_transfer_tensor(
            memory_obj, group_ctx
        )

        slot_mapping = slot_mapping.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        for layer_pos, layer_idx in enumerate(group_ctx.layer_indices):
            key_view, value_view = self._get_attention_kv_views(
                self.kvcaches[layer_idx],
                group_ctx,
            )
            selected_k = key_view.index_select(0, slot_mapping)
            selected_v = value_view.index_select(0, slot_mapping)
            memory_obj_tensor[0, layer_pos].copy_(selected_k, non_blocking=True)
            memory_obj_tensor[1, layer_pos].copy_(selected_v, non_blocking=True)
            if _should_log_tensor_samples(layer_pos, len(group_ctx.layer_indices)):
                sampled_rows = _build_edge_indices(selected_k.shape[0], count=2)
                sampled_slots = _cpu_select_rows(slot_mapping, sampled_rows)
                sampled_source_k = _cpu_select_rows(selected_k, sampled_rows)
                sampled_source_v = _cpu_select_rows(selected_v, sampled_rows)
                sampled_dest_k = _cpu_select_rows(memory_obj_tensor[0, layer_pos], sampled_rows)
                sampled_dest_v = _cpu_select_rows(memory_obj_tensor[1, layer_pos], sampled_rows)
                logger.info(
                    "NPU V3 attention store sample req_id=%s group=%d layer_idx=%d "
                    "layer_pos=%d slot_rows=%s slot_values=%s source_k=%s "
                    "dest_k=%s source_v=%s dest_v=%s",
                    req_id,
                    group_ctx.group_idx,
                    layer_idx,
                    layer_pos,
                    sampled_rows,
                    _summarize_int_values(
                        [int(v) for v in sampled_slots.tolist()]
                    ),
                    _tensor_sample_summary(sampled_source_k),
                    _tensor_sample_summary(sampled_dest_k),
                    _tensor_sample_summary(sampled_source_v),
                    _tensor_sample_summary(sampled_dest_v),
                )

    def _copy_attention_group_to_gpu_python(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        slot_mapping: torch.Tensor,
        req_id: Optional[str] = None,
    ) -> None:
        assert self.kvcaches is not None
        memory_obj_tensor = self._get_attention_group_transfer_tensor(
            memory_obj, group_ctx
        )

        slot_mapping = slot_mapping.to(
            device=self.device,
            dtype=torch.long,
            non_blocking=True,
        )

        for layer_pos, layer_idx in enumerate(group_ctx.layer_indices):
            key_view, value_view = self._get_attention_kv_views(
                self.kvcaches[layer_idx],
                group_ctx,
            )
            source_k = memory_obj_tensor[0, layer_pos].to(
                device=key_view.device,
                dtype=key_view.dtype,
                non_blocking=True,
            )
            source_v = memory_obj_tensor[1, layer_pos].to(
                device=value_view.device,
                dtype=value_view.dtype,
                non_blocking=True,
            )
            key_view.index_copy_(0, slot_mapping, source_k)
            value_view.index_copy_(0, slot_mapping, source_v)
            if _should_log_tensor_samples(layer_pos, len(group_ctx.layer_indices)):
                sampled_rows = _build_edge_indices(source_k.shape[0], count=2)
                sampled_slots = _cpu_select_rows(slot_mapping, sampled_rows)
                sampled_source_k = _cpu_select_rows(source_k, sampled_rows)
                sampled_source_v = _cpu_select_rows(source_v, sampled_rows)
                sampled_slot_values = [int(v) for v in sampled_slots.tolist()]
                sampled_dest_k = _cpu_select_rows(key_view, sampled_slot_values)
                sampled_dest_v = _cpu_select_rows(value_view, sampled_slot_values)
                logger.info(
                    "NPU V3 attention load sample req_id=%s group=%d layer_idx=%d "
                    "layer_pos=%d slot_rows=%s slot_values=%s source_k=%s "
                    "dest_k=%s source_v=%s dest_v=%s",
                    req_id,
                    group_ctx.group_idx,
                    layer_idx,
                    layer_pos,
                    sampled_rows,
                    _summarize_int_values(sampled_slot_values),
                    _tensor_sample_summary(sampled_source_k),
                    _tensor_sample_summary(sampled_dest_k),
                    _tensor_sample_summary(sampled_source_v),
                    _tensor_sample_summary(sampled_dest_v),
                )

    def _get_gdn_state_block_index(
        self,
        end: int,
        block_size: int,
        caller_name: str,
    ) -> int:
        if block_size <= 0:
            raise ValueError(
                f"{caller_name} requires a positive GDN block_size, got {block_size}."
            )
        if end <= 0:
            raise ValueError(f"{caller_name} requires end > 0, got {end}.")
        if end % block_size != 0:
            raise NotImplementedError(
                f"{caller_name} only supports GDN chunk ends aligned to the block size. "
                f"end={end}, block_size={block_size}"
            )
        return end // block_size - 1

    def _resolve_gdn_block_id(
        self,
        group_ctx: _NPUV3GroupContext,
        end: int,
        kwargs,
        caller_name: str,
    ) -> int:
        block_ids_by_group = _get_block_ids_by_group_from_kwargs(kwargs, caller_name)
        if group_ctx.group_idx >= len(block_ids_by_group):
            raise ValueError(
                f"{caller_name} received {len(block_ids_by_group)} block-id groups, "
                f"but needs group index {group_ctx.group_idx}."
            )
        state_block_index = self._get_gdn_state_block_index(
            end,
            group_ctx.block_size,
            caller_name,
        )
        group_block_ids = block_ids_by_group[group_ctx.group_idx]
        if state_block_index >= len(group_block_ids):
            raise ValueError(
                f"{caller_name} could not resolve GDN state block index {state_block_index} "
                f"for group {group_ctx.group_idx}; only {len(group_block_ids)} block ids "
                "are available."
            )
        return group_block_ids[state_block_index]

    def _collect_gdn_group_runtime_state_tensors(
        self,
        group_ctx: _NPUV3GroupContext,
    ) -> List[torch.Tensor]:
        assert self.kvcaches is not None

        group_kvcaches = [
            self.kvcaches[layer_idx]
            for layer_idx in group_ctx.layer_indices
        ]

        collected_state_tensors: List[torch.Tensor] = []
        for tensor_idx in range(group_ctx.num_tensors):
            for layer_pos, layer_cache in enumerate(group_kvcaches):
                if not isinstance(layer_cache, (tuple, list)):
                    raise ValueError(
                        f"GDN group {group_ctx.group_idx} expects per-layer state tensors, "
                        f"got {type(layer_cache)}."
                    )
                if len(layer_cache) != group_ctx.num_tensors:
                    raise ValueError(
                        f"GDN group {group_ctx.group_idx} expects {group_ctx.num_tensors} "
                        f"runtime tensors, got {len(layer_cache)}."
                    )

                state_tensor = layer_cache[tensor_idx]
                if not isinstance(state_tensor, torch.Tensor):
                    raise ValueError(
                        f"GDN group {group_ctx.group_idx} expects runtime tensor, "
                        f"got {type(state_tensor)} at layer_pos={layer_pos}, "
                        f"tensor_idx={tensor_idx}."
                    )
                collected_state_tensors.append(state_tensor)

        return collected_state_tensors

    def _validate_qwen3_5_gdn_group_contract(
        self,
        group_ctx: _NPUV3GroupContext,
        memory_tensors: List[torch.Tensor],
        state_tensors: List[torch.Tensor],
    ) -> None:
        if group_ctx.group_kind != "gdn":
            raise ValueError(
                f"GDN operator path requires group_kind='gdn', got {group_ctx.group_kind}."
            )
        if group_ctx.kv_format != KVCacheFormat.GDN_ALIGN_STATE:
            raise ValueError(
                f"GDN operator path requires kv_format=GDN_ALIGN_STATE, "
                f"got {group_ctx.kv_format.name}."
            )
        if group_ctx.num_tensors != 2:
            raise ValueError(
                f"First-version GDN operator only supports 2 tensors, "
                f"got {group_ctx.num_tensors}."
            )
        if group_ctx.tensor_names != ["conv_state", "ssm_state"]:
            raise ValueError(
                "First-version GDN operator only supports tensor_names="
                "['conv_state', 'ssm_state'], got "
                f"{group_ctx.tensor_names}."
            )
        if len(memory_tensors) != 2:
            raise ValueError(
                f"First-version GDN operator expects 2 memory tensors, "
                f"got {len(memory_tensors)}."
            )

        num_layers = group_ctx.num_layers
        expected_state_tensor_count = num_layers * 2
        if len(state_tensors) != expected_state_tensor_count:
            raise ValueError(
                f"First-version GDN operator expects {expected_state_tensor_count} "
                f"runtime state tensors, got {len(state_tensors)}."
            )

        conv_memory_tensor, ssm_memory_tensor = memory_tensors
        if not isinstance(conv_memory_tensor, torch.Tensor) or not isinstance(
            ssm_memory_tensor, torch.Tensor
        ):
            raise ValueError("GDN operator path expects memory tensors to be torch.Tensor.")

        if conv_memory_tensor.dtype != torch.bfloat16:
            raise ValueError(
                "First-version GDN operator expects conv_state memory tensor dtype "
                f"torch.bfloat16, got {conv_memory_tensor.dtype}."
            )
        if ssm_memory_tensor.dtype != torch.float32:
            raise ValueError(
                "First-version GDN operator expects ssm_state memory tensor dtype "
                f"torch.float32, got {ssm_memory_tensor.dtype}."
            )

        if conv_memory_tensor.shape[0] != num_layers:
            raise ValueError(
                f"conv_state memory tensor expects first dim {num_layers}, "
                f"got {conv_memory_tensor.shape[0]}."
            )
        if ssm_memory_tensor.shape[0] != num_layers:
            raise ValueError(
                f"ssm_state memory tensor expects first dim {num_layers}, "
                f"got {ssm_memory_tensor.shape[0]}."
            )

        conv_state_tensors = state_tensors[:num_layers]
        ssm_state_tensors = state_tensors[num_layers:]

        for layer_pos, state_tensor in enumerate(conv_state_tensors):
            if state_tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"conv_state runtime tensor at layer_pos={layer_pos} expects "
                    f"torch.bfloat16, got {state_tensor.dtype}."
                )
            if tuple(state_tensor.shape[1:]) != tuple(conv_memory_tensor.shape[1:]):
                raise ValueError(
                    f"conv_state runtime tensor tail shape {tuple(state_tensor.shape[1:])} "
                    f"does not match transfer tail shape {tuple(conv_memory_tensor.shape[1:])}."
                )

        for layer_pos, state_tensor in enumerate(ssm_state_tensors):
            if state_tensor.dtype != torch.float32:
                raise ValueError(
                    f"ssm_state runtime tensor at layer_pos={layer_pos} expects "
                    f"torch.float32, got {state_tensor.dtype}."
                )
            if tuple(state_tensor.shape[1:]) != tuple(ssm_memory_tensor.shape[1:]):
                raise ValueError(
                    f"ssm_state runtime tensor tail shape {tuple(state_tensor.shape[1:])} "
                    f"does not match transfer tail shape {tuple(ssm_memory_tensor.shape[1:])}."
                )

    def _run_gdn_group_from_gpu_op(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        if not hasattr(lmc_ops, "multi_layer_gdn_state_transfer"):
            raise NotImplementedError(
                "multi_layer_gdn_state_transfer is not available in lmcache_ascend.c_ops."
            )

        block_id = self._resolve_gdn_block_id(
            group_ctx,
            end,
            kwargs,
            "VLLMPagedMemNPUConnectorV3.from_gpu",
        )
        group_memory_tensors = self._get_group_memory_tensors(memory_obj, group_ctx)
        state_tensors = self._collect_gdn_group_runtime_state_tensors(group_ctx)
        self._validate_qwen3_5_gdn_group_contract(
            group_ctx,
            group_memory_tensors,
            state_tensors,
        )

        lmc_ops.multi_layer_gdn_state_transfer(
            group_memory_tensors,
            state_tensors,
            block_id,
            True,
        )

    def _run_gdn_group_to_gpu_op(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        if not hasattr(lmc_ops, "multi_layer_gdn_state_transfer"):
            raise NotImplementedError(
                "multi_layer_gdn_state_transfer is not available in lmcache_ascend.c_ops."
            )

        block_id = self._resolve_gdn_block_id(
            group_ctx,
            end,
            kwargs,
            "VLLMPagedMemNPUConnectorV3.to_gpu",
        )
        group_memory_tensors = self._get_group_memory_tensors(memory_obj, group_ctx)
        state_tensors = self._collect_gdn_group_runtime_state_tensors(group_ctx)
        self._validate_qwen3_5_gdn_group_contract(
            group_ctx,
            group_memory_tensors,
            state_tensors,
        )

        lmc_ops.multi_layer_gdn_state_transfer(
            group_memory_tensors,
            state_tensors,
            block_id,
            False,
        )

    def _copy_gdn_group_from_gpu_python(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        assert self.kvcaches is not None

        # 获得当前组所在的 blockid
        block_id = self._resolve_gdn_block_id(
            group_ctx,
            end,
            kwargs,
            "VLLMPagedMemNPUConnectorV3.from_gpu",
        )

        # memory_tensor_conv.shape = [12, 3, 8192]
        # memory_tensor_ssm.shape = [12, 32, 128, 128]
        group_memory_tensors = self._get_group_memory_tensors(memory_obj, group_ctx)
        if len(group_memory_tensors) != group_ctx.num_tensors:
            raise ValueError(
                f"GDN group {group_ctx.group_idx} expects {group_ctx.num_tensors} tensors, "
                f"but memory object exposes {len(group_memory_tensors)}."
            )

        #获取当前组所有层的数据 当前 GDN group 的所有层的 runtime cache 列表 每个 layer_cache 对应这个 group 里的 一层
        group_kvcaches = [
            self.kvcaches[layer_idx]
            for layer_idx in group_ctx.layer_indices
        ]
        for layer_pos, layer_cache in enumerate(group_kvcaches):
            if not isinstance(layer_cache, (tuple, list)):
                raise ValueError(
                    f"GDN group {group_ctx.group_idx} expects per-layer state tensors, "
                    f"got {type(layer_cache)}."
                )
            if len(layer_cache) != group_ctx.num_tensors:
                raise ValueError(
                    f"GDN group {group_ctx.group_idx} expects {group_ctx.num_tensors} "
                    f"runtime tensors, got {len(layer_cache)}."
                )
            for tensor_idx, (memory_tensor, state_tensor) in enumerate(
                zip(group_memory_tensors, layer_cache, strict=True)
            ):
                memory_tensor[layer_pos].copy_(state_tensor[block_id])
                if _should_log_tensor_samples(layer_pos, len(group_kvcaches)):
                    source_state = state_tensor[block_id]
                    dest_state = memory_tensor[layer_pos]
                    logger.info(
                        "NPU V3 gdn store sample req_id=%s group=%d layer_pos=%d "
                        "layer_idx=%d block_id=%d tensor_idx=%d tensor_name=%s "
                        "source=%s dest=%s",
                        kwargs.get("req_id"),
                        group_ctx.group_idx,
                        layer_pos,
                        group_ctx.layer_indices[layer_pos],
                        block_id,
                        tensor_idx,
                        group_ctx.tensor_names[tensor_idx],
                        _tensor_sample_summary(source_state),
                        _tensor_sample_summary(dest_state),
                    )

    def _copy_gdn_group_to_gpu_python(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        assert self.kvcaches is not None
        block_id = self._resolve_gdn_block_id(
            group_ctx,
            end,
            kwargs,
            "VLLMPagedMemNPUConnectorV3.to_gpu",
        )
        group_memory_tensors = self._get_group_memory_tensors(memory_obj, group_ctx)
        if len(group_memory_tensors) != group_ctx.num_tensors:
            raise ValueError(
                f"GDN group {group_ctx.group_idx} expects {group_ctx.num_tensors} tensors, "
                f"but memory object exposes {len(group_memory_tensors)}."
            )

        group_kvcaches = [
            self.kvcaches[layer_idx]
            for layer_idx in group_ctx.layer_indices
        ]
        for layer_pos, layer_cache in enumerate(group_kvcaches):
            if not isinstance(layer_cache, (tuple, list)):
                raise ValueError(
                    f"GDN group {group_ctx.group_idx} expects per-layer state tensors, "
                    f"got {type(layer_cache)}."
                )
            if len(layer_cache) != group_ctx.num_tensors:
                raise ValueError(
                    f"GDN group {group_ctx.group_idx} expects {group_ctx.num_tensors} "
                    f"runtime tensors, got {len(layer_cache)}."
                )
            for tensor_idx, (memory_tensor, state_tensor) in enumerate(zip(
                group_memory_tensors,
                layer_cache,
                strict=True,
            )):
                state_tensor[block_id].copy_(memory_tensor[layer_pos])
                if _should_log_tensor_samples(layer_pos, len(group_kvcaches)):
                    source_state = memory_tensor[layer_pos]
                    dest_state = state_tensor[block_id]
                    logger.info(
                        "NPU V3 gdn load sample req_id=%s group=%d layer_pos=%d "
                        "layer_idx=%d block_id=%d tensor_idx=%d tensor_name=%s "
                        "source=%s dest=%s",
                        kwargs.get("req_id"),
                        group_ctx.group_idx,
                        layer_pos,
                        group_ctx.layer_indices[layer_pos],
                        block_id,
                        tensor_idx,
                        group_ctx.tensor_names[tensor_idx],
                        _tensor_sample_summary(source_state),
                        _tensor_sample_summary(dest_state),
                    )

    def _copy_gdn_group_from_gpu(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        req_id = kwargs.get("req_id")
        try:
            self._run_gdn_group_from_gpu_op(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )
        except NotImplementedError as exc:
            if not self._gdn_op_unavailable_warned:
                logger.warning(
                    "GDN operator path is unavailable, falling back to Python copy. "
                    "req_id=%s group=%d error=%s",
                    req_id,
                    group_ctx.group_idx,
                    exc,
                )
                self._gdn_op_unavailable_warned = True
            self._copy_gdn_group_from_gpu_python(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "GDN operator store failed, falling back to Python copy. "
                "req_id=%s group=%d error=%s",
                req_id,
                group_ctx.group_idx,
                exc,
            )
            self._copy_gdn_group_from_gpu_python(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )

    def _copy_gdn_group_to_gpu(
        self,
        memory_obj: MemoryObj,
        group_ctx: _NPUV3GroupContext,
        end: int,
        **kwargs,
    ) -> None:
        req_id = kwargs.get("req_id")
        gdn_align_last_end = kwargs.get("gdn_align_last_end")
        if (
            _GDN_ALIGN_LOAD_LAST_ONLY
            and group_ctx.kv_format == KVCacheFormat.GDN_ALIGN_STATE
            and gdn_align_last_end is not None
            and end != gdn_align_last_end
        ):
            logger.info(
                "Skip GDN align load for non-last chunk req_id=%s group=%d "
                "end=%d last_end=%d",
                req_id,
                group_ctx.group_idx,
                end,
                gdn_align_last_end,
            )
            return
        try:
            self._run_gdn_group_to_gpu_op(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )
        except NotImplementedError as exc:
            if not self._gdn_op_unavailable_warned:
                logger.warning(
                    "GDN operator path is unavailable, falling back to Python copy. "
                    "req_id=%s group=%d error=%s",
                    req_id,
                    group_ctx.group_idx,
                    exc,
                )
                self._gdn_op_unavailable_warned = True
            self._copy_gdn_group_to_gpu_python(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "GDN operator load failed, falling back to Python copy. "
                "req_id=%s group=%d error=%s",
                req_id,
                group_ctx.group_idx,
                exc,
            )
            self._copy_gdn_group_to_gpu_python(
                memory_obj,
                group_ctx,
                end,
                **kwargs,
            )

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        req_id = kwargs.get("req_id")
        slot_mappings_by_group = self._ensure_group_slot_mappings(
            kwargs, "VLLMPagedMemNPUConnectorV3.to_gpu"
        )
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        self._initialize_group_contexts()
        assert self.group_contexts is not None

        for group_ctx, slot_mapping in zip(
            self.group_contexts, slot_mappings_by_group, strict=True
        ):
            if group_ctx.kv_format == KVCacheFormat.GDN_ALIGN_STATE:
                self._copy_gdn_group_to_gpu(memory_obj, group_ctx, end, **kwargs)
                continue
            self._copy_attention_group_to_gpu(
                memory_obj,
                group_ctx,
                slot_mapping[start:end],
                req_id=req_id,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        req_id = kwargs.get("req_id")
        slot_mappings_by_group = self._ensure_group_slot_mappings(
            kwargs, "VLLMPagedMemNPUConnectorV3.from_gpu"
        )
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        self._initialize_group_contexts()
        assert self.group_contexts is not None

        with torch.cuda.stream(self.store_stream):
            for group_ctx, slot_mapping in zip( #按组进行遍历
                self.group_contexts, slot_mappings_by_group, strict=True
            ):
                if group_ctx.kv_format == KVCacheFormat.GDN_ALIGN_STATE:
                    self._copy_gdn_group_from_gpu(memory_obj, group_ctx, end, **kwargs)
                    continue
                self._copy_attention_group_from_gpu(
                    memory_obj,
                    group_ctx,
                    slot_mapping[start:end],
                    req_id=req_id,
                )

        if not memory_obj.raw_tensor.is_cuda:
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        slot_mappings_by_group = self._ensure_group_slot_mappings(
            kwargs, "VLLMPagedMemNPUConnectorV3.batched_to_gpu"
        )
        if any(getattr(m, "is_proxy", False) for m in memory_objs):
            raise NotImplementedError(
                "VLLMPagedMemNPUConnectorV3 first version does not support "
                "ProxyMemoryObj / remote batched_to_gpu."
            )

        gdn_align_last_end = max(ends) if ends else None
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(
                    memory_obj,
                    start,
                    end,
                    slot_mappings_by_group=slot_mappings_by_group,
                    gdn_align_last_end=gdn_align_last_end,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in (
                            "slot_mappings_by_group",
                            "slot_mapping",
                            "gdn_align_last_end",
                        )
                    },
                )
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        slot_mappings_by_group = self._ensure_group_slot_mappings(
            kwargs, "VLLMPagedMemNPUConnectorV3.batched_from_gpu"
        )
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(
                memory_obj,
                start,
                end,
                slot_mappings_by_group=slot_mappings_by_group,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("slot_mappings_by_group", "slot_mapping")
                },
            )

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError(
            "VLLMPagedMemNPUConnectorV3 uses metadata.get_shapes() per KV group."
        )


class VLLMPagedMemLayerwiseNPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)

        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

        # layerwise mode currently does not support MLA
        self.use_mla = kwargs.get("use_mla", False)

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.

        Supports both legacy formats and new SEPARATE_KV format:
        - Legacy MERGED_KV: [2, num_blocks, block_size, num_heads, head_size]
        - New SEPARATE_KV: tuple(key_tensor, value_tensor) where each is
          [num_blocks, block_size, num_heads, head_size]
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")

            self.kv_format = KVCacheFormat.detect(kv_caches, use_mla=self.use_mla)

            if self.kv_format == KVCacheFormat.UNDEFINED:
                raise ValueError(
                    "Undefined KV cache format detected. "
                    "Unable to determine the format of input kv_caches."
                )

            logger.info(f"Detected KV cache format: {self.kv_format.name}")

            first_layer_cache = kv_caches[0]

            if self.kv_format == KVCacheFormat.SEPARATE_KV:
                key_tensor = first_layer_cache[0]
                value_tensor = first_layer_cache[1]

                assert key_tensor.shape == value_tensor.shape, (
                    f"Key and Value tensors must have identical shapes, "
                    f"got key={key_tensor.shape}, value={value_tensor.shape}"
                )

                k_cache_shape_per_layer = key_tensor.shape
                self.vllm_two_major = False

            elif self.kv_format == KVCacheFormat.MERGED_KV:
                assert (
                    first_layer_cache.shape[0] == 2 or first_layer_cache.shape[1] == 2
                ), (
                    "MERGED_KV format should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                    f"Got shape: {first_layer_cache.shape}"
                )

                self.vllm_two_major = first_layer_cache.shape[0] == 2

                if self.vllm_two_major:
                    # Flash Attention: [2, num_blocks, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[0].shape
                else:
                    # Flash Infer: [num_blocks, 2, block_size, num_heads, head_size]
                    k_cache_shape_per_layer = first_layer_cache[:, 0].shape
            else:
                raise ValueError(f"Unsupported KV cache format: {self.kv_format}")

            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}"
            )

            num_elements_key = k_cache_shape_per_layer.numel()
            num_elements = num_elements_key * 2
            gpu_buffer_size = num_elements * self.element_size

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemLayerwiseNPUConnector.batched_to_gpu"
        )

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")
            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                if self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: N H2D memcpy + 1 scatter kernel
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,  # CPU memory objects
                        tmp_gpu_buffer_obj.tensor,  # GPU staging buffer
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,  # offset for each chunk
                        chunk_sizes,  # size for each chunk
                        False,  # to_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )

                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            False,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )
                logger.debug(f"Finished loading layer {layer_id}")
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        # if "slot_mapping" not in kwargs:
        #     raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        # slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slot_mapping = _get_legacy_single_slot_mapping_from_kwargs(
            kwargs, "VLLMPagedMemLayerwiseNPUConnector.batched_from_gpu"
        )
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        chunk_offsets = []
        chunk_sizes = []
        current_offset = 0

        for start, end in zip(starts, ends, strict=False):
            chunk_size = end - start
            chunk_sizes.append(chunk_size)
            chunk_offsets.append(current_offset)
            current_offset += chunk_size

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate NPU buffer in NPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)

                if self.use_gpu:
                    cpu_tensors = []
                    for memory_obj in memory_objs_layer:
                        assert memory_obj.tensor is not None
                        cpu_tensors.append(memory_obj.tensor)

                    # Fused transfer: 1 scatter kernel + N D2H memcpy
                    lmc_ops.batched_fused_single_layer_kv_transfer(
                        cpu_tensors,
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        chunk_offsets,
                        chunk_sizes,
                        True,  # from_gpu
                        self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                        True,  # token_major
                        self.vllm_two_major,
                    )
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.tensor is not None

                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            self.kv_format.value,  # 1:MERGED_KV / 2:SEPARATE_KV
                            True,
                            self.vllm_two_major,
                        )
                logger.debug(f"Finished offloading layer {layer_id}")
            yield

            if sync:
                self.store_stream.synchronize()

        # free the buffer memory
        if self.use_gpu and tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()
        yield


class SGLangNPUConnector(SGLangGPUConnector):
    pass


class SGLangLayerwiseNPUConnector(SGLangLayerwiseGPUConnector):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [num_blocks, block_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        super().__init__(hidden_dim_size, num_layers, use_gpu, **kwargs)
        self.kv_format: KVCacheFormat = KVCacheFormat.UNDEFINED

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        # [2, self.layer_num, self.size // self.page_size + 1,
        # self.page_size, self.head_num, self.head_dim,]
        self.kv_format = KVCacheFormat.detect(kv_caches)
        if self.kv_format == KVCacheFormat.UNDEFINED:
            raise ValueError("Could not detect KV cache format.")

        if self.use_gpu and self.gpu_buffer_allocator is None:
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size

            logger.info(
                f"Lazily initializing GPU buffer:\n"
                f"  - Format: {self.kv_format.name}\n"
                f"  - Key cache shape per layer: {k_cache_shape_per_layer}\n"
                f"  - Max tokens: {max_tokens}\n"
                f"  - num_elements: {num_elements}\n"
                f"  - gpu_buffer_size: {gpu_buffer_size / (1024 * 1024)} MB"
            )

            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        False,
                        self.kv_format.value,
                        True,
                        True,
                    )

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    False,
                    self.kv_format.value,
                    True,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            current_layer_kv = (self.kvcaches[0][layer_id], self.kvcaches[1][layer_id])

            if self.use_gpu:
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    current_layer_kv,
                    slot_mapping_full,
                    True,
                    self.kv_format.value,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None

                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer(
                        memory_obj.tensor,
                        current_layer_kv,
                        slot_mapping[start:end],
                        True,
                        self.kv_format.value,
                        True,
                        True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
