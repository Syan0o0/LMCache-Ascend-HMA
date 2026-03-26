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


def patched_hidden_dim_size(self) -> int:
    """Return the size of the hidden dimension in this group."""
    # hidden_dim_size = num_heads * head_size
    if getattr(self, "group_kind", KVLayerGroupKind.ATTENTION) == KVLayerGroupKind.GDN:
        primary_shape = self.tensor_specs[0].shape
        if len(primary_shape) >= 2:
            return primary_shape[-1]
        raise ValueError(f"Invalid GDN shape for hidden dim size: {primary_shape}")
    if len(self.shape) == 5:
        # MHA
        return self.shape[3] * self.shape[4]
    elif len(self.shape) == 4:
        # NOTE(gingfung): Ascend separated format for KVCaches
        # i.e. a tuple of kv (numblocks, blocksize, heads, headdim)
        #      very unlikely, but potentially MLA with (1, ....)
        if self.shape[0] == 1:
            raise ValueError(f"Invalid shape for hidden dim size: {self.shape}")

        return self.shape[2] * self.shape[3]
    elif len(self.shape) == 3:
        # MLA
        return self.shape[2]
    else:
        raise ValueError(f"Invalid shape: {self.shape}")


def _infer_group_kind_and_tensor_specs(
    kv_cache: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[KVLayerGroupKind, list[KVLayerTensorSpec]]:
    if isinstance(kv_cache, torch.Tensor):
        return (
            KVLayerGroupKind.ATTENTION,
            [KVLayerTensorSpec(name="kv", shape=kv_cache.shape, dtype=kv_cache.dtype)],
        )

    if not isinstance(kv_cache, Sequence) or len(kv_cache) == 0:
        raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")

    tensors = list(kv_cache)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError(f"Unknown KVCache type: {type(kv_cache)}")

    is_attention_pair = (
        len(tensors) == 2
        and tensors[0].shape == tensors[1].shape
        and tensors[0].dtype == tensors[1].dtype
        and tensors[0].ndim == 4
    )
    if is_attention_pair:
        names = ["k", "v"]
        group_kind = KVLayerGroupKind.ATTENTION
    else:
        names = ["conv_state", "ssm_state"] if len(tensors) == 2 else [
            f"tensor_{idx}" for idx in range(len(tensors))
        ]
        group_kind = KVLayerGroupKind.GDN

    return (
        group_kind,
        [
            KVLayerTensorSpec(name=name, shape=tensor.shape, dtype=tensor.dtype)
            for name, tensor in zip(names, tensors, strict=True)
        ],
    )


def build_kv_layer_groups(
    self,
    kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
) -> None:
    """Build KV layer groups structure by analyzing each layer's shape and dtype.

    Layers with the same shape and dtype are grouped together. This is useful
    because different layers may have different structures (especially the
    last dimension head_size may differ between groups), and different groups
    may have different dtypes.

    If layer groups are already built (non-empty list), this method does nothing.

    Args:
        kv_caches: Dictionary mapping layer names to KV cache tensors.
    """
    # Skip if already built (non-empty list)
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    groups_dict: dict[
        tuple[
            KVLayerGroupKind,
            tuple[tuple[str, torch.Size, torch.dtype], ...],
        ],
        list[tuple[str, int]],
    ] = defaultdict(list)
    group_specs: dict[
        tuple[
            KVLayerGroupKind,
            tuple[tuple[str, torch.Size, torch.dtype], ...],
        ],
        list[KVLayerTensorSpec],
    ] = {}

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        logger.debug("KVCache Type: %s", type(kv_cache))
        group_kind, tensor_specs = _infer_group_kind_and_tensor_specs(kv_cache)
        key = (
            group_kind,
            tuple(
                (tensor_spec.name, tensor_spec.shape, tensor_spec.dtype)
                for tensor_spec in tensor_specs
            ),
        )
        group_specs[key] = tensor_specs
        groups_dict[key].append((layer_name, idx))

    # Build KVLayerGroupInfo list
    # Sort groups by the first layer index to maintain order
    def _get_first_layer_index(shape_dtype_key):
        """Get the index of the first layer in a layer group."""
        layer_group = groups_dict[
            shape_dtype_key
        ]  # list of (layer_name, layer_index) tuples
        first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
        layer_index = first_layer_info[1]  # extract the layer index
        return layer_index

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for key in sorted_keys:
        layers = groups_dict[key]
        layer_names, layer_indices = zip(*layers, strict=False)
        tensor_specs = group_specs[key]

        group_info = KVLayerGroupInfo(
            layer_names=list(layer_names),
            layer_indices=list(layer_indices),
            shape=tensor_specs[0].shape,
            dtype=tensor_specs[0].dtype,
            group_kind=key[0],
            tensor_specs=tensor_specs,
        )
        kv_layer_groups.append(group_info)

    # Store the built groups
    self.kv_layer_groups = kv_layer_groups

    # Print the group structure
    logger.info("KV layer groups: %s", kv_layer_groups)
