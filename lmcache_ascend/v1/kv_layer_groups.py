# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Sequence

# Third Party
from lmcache.logging import init_logger
from lmcache.v1.kv_layer_groups import (
    KVLayerGroupInfo,
    KVLayerGroupKind,
    KVLayerTensorSpec,
)
import torch

logger = init_logger(__name__)


def _get_tuple_storage_shape(kv_cache: Sequence[torch.Tensor]) -> torch.Size:
    """Return the LMCache storage shape for tuple-based attention KV caches.

    For separated K/V, the storage shape is the same as a single K/V tensor.
    For MLA/DSA, LMCache stores multiple KV tensors as one contiguous hidden
    dimension, so the storage shape must reflect the flattened hidden size.
    """
    first_shape = kv_cache[0].shape

    for tensor in kv_cache[1:]:
        if tensor.shape[:2] != first_shape[:2]:
            raise ValueError(
                "All attention KV tensors in a tuple must share "
                "[num_blocks, block_size], "
                f"got {first_shape} and {tensor.shape}"
            )

    if len(kv_cache) == 2 and kv_cache[0].shape == kv_cache[1].shape:
        return first_shape

    total_hidden_dim = sum(tensor.shape[-2] * tensor.shape[-1] for tensor in kv_cache)
    return torch.Size([first_shape[0], first_shape[1], total_hidden_dim])


def _is_attention_tuple_layout(tensors: Sequence[torch.Tensor]) -> bool:
    """Detect tuple-based attention layouts used by Ascend runtimes.

    Attention tuple layouts share the paged-KV prefix dimensions
    ``[num_blocks, block_size]`` across all tensors. This covers:
    - separated K/V with matching K/V shapes
    - MLA with 2 tensors whose last dimensions differ
    - DSA with 3 tensors (k, v, dsa_k)

    GDN state groups do not follow this paged-KV prefix layout.
    """
    if len(tensors) < 2:
        return False

    first = tensors[0]
    if first.ndim < 4:
        return False

    kv_prefix = first.shape[:2]
    return all(tensor.ndim >= 4 and tensor.shape[:2] == kv_prefix for tensor in tensors)


def _get_attention_tensor_names(tensors: Sequence[torch.Tensor]) -> list[str]:
    if len(tensors) == 2:
        return ["k", "v"]
    if len(tensors) == 3:
        return ["k", "v", "dsa_k"]
    return [f"tensor_{idx}" for idx in range(len(tensors))]


def _infer_group_layout(
    kv_cache: torch.Tensor | Sequence[torch.Tensor],
) -> tuple[KVLayerGroupKind, list[KVLayerTensorSpec], torch.Size, torch.dtype]:
    """Infer Ascend KV group semantics while preserving tuple attention formats.

    Supported runtime KV cache forms:
    - Single-tensor format: one tensor, typically shaped like
      [2, num_blocks, block_size, num_heads, head_size].
    - Tuple/list attention formats: multiple tensors sharing the paged-KV
      prefix [num_blocks, block_size]. This includes SEPARATE_KV, MLA_KV,
      and DSA_KV layouts.
    - Other multi-tensor layouts are treated as GDN/state groups.
    """
    if isinstance(kv_cache, torch.Tensor):
        tensor_spec = KVLayerTensorSpec(
            name="kv",
            shape=kv_cache.shape,
            dtype=kv_cache.dtype,
        )
        return (
            KVLayerGroupKind.ATTENTION,
            [tensor_spec],
            kv_cache.shape,
            kv_cache.dtype,
        )

    if not isinstance(kv_cache, Sequence) or len(kv_cache) == 0:
        raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")

    tensors = list(kv_cache)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError(f"Unknown KVCache element type: {type(kv_cache)}")

    if _is_attention_tuple_layout(tensors):
        tensor_specs = [
            KVLayerTensorSpec(name=name, shape=tensor.shape, dtype=tensor.dtype)
            for name, tensor in zip(
                _get_attention_tensor_names(tensors), tensors, strict=True
            )
        ]
        return (
            KVLayerGroupKind.ATTENTION,
            tensor_specs,
            _get_tuple_storage_shape(tensors),
            tensor_specs[0].dtype,
        )

    names = ["conv_state", "ssm_state"] if len(tensors) == 2 else [
        f"tensor_{idx}" for idx in range(len(tensors))
    ]
    tensor_specs = [
        KVLayerTensorSpec(name=name, shape=tensor.shape, dtype=tensor.dtype)
        for name, tensor in zip(names, tensors, strict=True)
    ]
    return (
        KVLayerGroupKind.GDN,
        tensor_specs,
        tensor_specs[0].shape,
        tensor_specs[0].dtype,
    )


def patched_hidden_dim_size(self) -> int:
    """Return the size of the hidden dimension in this group."""
    if getattr(self, "group_kind", KVLayerGroupKind.ATTENTION) == KVLayerGroupKind.GDN:
        primary_shape = self.tensor_specs[0].shape
        if len(primary_shape) >= 2:
            return primary_shape[-1]
        raise ValueError(f"Invalid GDN shape for hidden dim size: {primary_shape}")

    # hidden_dim_size = num_heads * head_size
    if len(self.shape) == 5:
        # MHA
        return self.shape[3] * self.shape[4]
    if len(self.shape) == 4:
        # NOTE(gingfung): Ascend separated format for KVCaches
        # i.e. a tuple of kv (numblocks, blocksize, heads, headdim)
        #      very unlikely, but potentially MLA with (1, ....)
        if self.shape[0] == 1:
            raise ValueError(f"Invalid shape for hidden dim size: {self.shape}")
        return self.shape[2] * self.shape[3]
    if len(self.shape) == 3:
        # MLA / DSA flattened hidden dimension.
        return self.shape[2]
    raise ValueError(f"Invalid shape: {self.shape}")


def build_kv_layer_groups(
    self,
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
) -> None:
    """Build KV layer groups structure by analyzing each layer's layout.

    Layers with the same logical layout are grouped together. On Ascend this
    includes both standard attention KV layouts and tuple-based attention
    layouts such as SEPARATE_KV / MLA_KV / DSA_KV, plus HMA-specific GDN
    state groups.

    If layer groups are already built (non-empty list), this method does
    nothing.

    Args:
        kv_caches: Dictionary mapping layer names to KV cache tensors.
    """
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    # Group layers by logical layout in a single loop.
    groups_dict: dict[
        tuple[
            KVLayerGroupKind,
            tuple[tuple[str, torch.Size, torch.dtype], ...],
        ],
        list[tuple[str, int]],
    ] = defaultdict(list)
    group_infos: dict[
        tuple[
            KVLayerGroupKind,
            tuple[tuple[str, torch.Size, torch.dtype], ...],
        ],
        tuple[torch.Size, torch.dtype, list[KVLayerTensorSpec]],
    ] = {}

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        group_kind, tensor_specs, shape, dtype = _infer_group_layout(kv_cache)
        key = (
            group_kind,
            tuple(
                (tensor_spec.name, tensor_spec.shape, tensor_spec.dtype)
                for tensor_spec in tensor_specs
            ),
        )
        groups_dict[key].append((layer_name, idx))
        group_infos[key] = (shape, dtype, tensor_specs)

    def _get_first_layer_index(group_key):
        """Get the index of the first layer in a layer group."""
        return groups_dict[group_key][0][1]

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for key in sorted_keys:
        shape, dtype, tensor_specs = group_infos[key]
        layers = groups_dict[key]
        layer_names, layer_indices = zip(*layers, strict=False)

        kv_layer_groups.append(
            KVLayerGroupInfo(
                layer_names=list(layer_names),
                layer_indices=list(layer_indices),
                shape=shape,
                dtype=dtype,
                group_kind=key[0],
                tensor_specs=tensor_specs,
            )
        )

    # Store the built groups
    self.kv_layer_groups = kv_layer_groups

    # Print the group structure
    logger.info("KV layer groups: %s", kv_layer_groups)
