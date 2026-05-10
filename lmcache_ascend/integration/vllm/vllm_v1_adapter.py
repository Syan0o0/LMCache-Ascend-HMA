# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
)
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupKind
from lmcache_ascend.v1.kv_layer_groups import (
    _infer_group_layout as _infer_ascend_group_layout,
)
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.v1.request import RequestStatus
import torch

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _get_request_slot_mappings_on_device(
    connector: "LMCacheAscendConnectorV1Impl",
    request: Any,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        slot_mapping.to(connector.device)
        for slot_mapping in request.slot_mappings_by_group
    )


def _assert_non_layerwise_hma_path(
    connector: "LMCacheAscendConnectorV1Impl",
    slot_mappings_by_group: tuple[torch.Tensor, ...],
    op_name: str,
) -> None:
    if len(slot_mappings_by_group) <= 1:
        return

    if connector.use_layerwise:
        raise NotImplementedError(
            f"{op_name} does not support multi-group KV cache with "
            "layerwise NPU connector yet."
        )

    if not connector.config.use_gpu_connector_v3:
        raise NotImplementedError(
            f"{op_name} received multi-group KV cache, but "
            "LMCache-Ascend is not using NPU connector V3 yet."
        )


class LMCacheAscendConnectorV1Impl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        logger.debug("Initializing LMCacheAscendConnectorV1Impl")
        super().__init__(vllm_config, role, parent)
        self._init_kv_group_config()
        self._sync_kv_group_metadata()
        self.store_async = self.config.store_async
        self._wait_for_save_done = True
        self._finished_req_ids_waiting_for_save: set[str] = set()
        self._late_finished_sending: set[str] = set()
        logger.debug("store_async: %s", self.store_async)

    def _init_kv_group_config(self) -> None:
        self._kv_cache_config = getattr(self._parent, "_kv_cache_config", None)
        if self._kv_cache_config is not None:
            self._num_kv_groups = len(self._kv_cache_config.kv_cache_groups)
            self._block_sizes_by_group = tuple(
                group.kv_cache_spec.block_size
                for group in self._kv_cache_config.kv_cache_groups
            )
        else:
            self._num_kv_groups = 1
            self._block_sizes_by_group = (self._block_size,)

        # Backward-compatible alias for legacy single-group code paths.
        self._block_size = self._block_sizes_by_group[0]

    def _sync_kv_group_metadata(self) -> None:
        if self.lmcache_engine is None:
            return
        self.lmcache_engine.metadata.kv_group_block_sizes = self._block_sizes_by_group

    def _build_vllm_aligned_kv_layer_groups(self) -> None:
        if self.lmcache_engine is None or self._kv_cache_config is None:
            return
        if len(self.kv_caches) == 0:
            return

        kv_layer_groups_manager = self.lmcache_engine.metadata.kv_layer_groups_manager
        kv_cache_items = list(self.kv_caches.items())
        layer_name_to_index = {
            layer_name: layer_idx
            for layer_idx, (layer_name, _) in enumerate(kv_cache_items)
        }

        kv_layer_groups: list[KVLayerGroupInfo] = []
        for group_idx, group_spec in enumerate(self._kv_cache_config.kv_cache_groups):
            group_layer_names: list[str] = []
            group_layer_indices: list[int] = []
            representative_cache = None

            for layer_name in group_spec.layer_names:
                if layer_name not in self.kv_caches:
                    logger.debug(
                        "Skipping layer %s while building vLLM-aligned KV group %d "
                        "because it is absent from registered kv_caches.",
                        layer_name,
                        group_idx,
                    )
                    continue

                group_layer_names.append(layer_name)
                group_layer_indices.append(layer_name_to_index[layer_name])
                if representative_cache is None:
                    representative_cache = self.kv_caches[layer_name]

            if representative_cache is None:
                raise ValueError(
                    "Failed to build vLLM-aligned KV layer groups because "
                    f"kv_cache_group[{group_idx}] with layers {group_spec.layer_names} "
                    "has no matching runtime kv caches."
                )

            group_kind, tensor_specs, shape, dtype = _infer_ascend_group_layout(
                representative_cache
            )
            kv_layer_groups.append(
                KVLayerGroupInfo(
                    layer_names=group_layer_names,
                    layer_indices=group_layer_indices,
                    shape=shape,
                    dtype=dtype,
                    group_kind=group_kind,
                    tensor_specs=tensor_specs,
                )
            )

        kv_layer_groups_manager.kv_layer_groups = kv_layer_groups

    def _validate_gdn_chunk_alignment(self) -> None:
        if self.lmcache_engine is None:
            return

        kv_layer_groups = (
            self.lmcache_engine.metadata.kv_layer_groups_manager.kv_layer_groups
        )
        if not kv_layer_groups:
            return

        has_gdn_group = False
        for group_idx, group in enumerate(kv_layer_groups):
            if group.group_kind != KVLayerGroupKind.GDN:
                continue
            has_gdn_group = True
            block_size = self._block_sizes_by_group[group_idx]
            if self._lmcache_chunk_size % block_size != 0:
                raise NotImplementedError(
                    "GDN align-state caching requires lmcache_chunk_size to be "
                    "a multiple of the GDN block size. "
                    f"chunk_size={self._lmcache_chunk_size}, block_size={block_size}, "
                    f"group_idx={group_idx}"
                )

        if has_gdn_group and self._discard_partial_chunks is False:
            raise NotImplementedError(
                "GDN align-state caching requires "
                "discard_partial_chunks=True / save_unfull_chunk=False."
            )

    def _refresh_hma_runtime_metadata(self) -> None:
        self._sync_kv_group_metadata()
        self._build_vllm_aligned_kv_layer_groups()
        self._validate_gdn_chunk_alignment()

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        self._refresh_hma_runtime_metadata()

    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        super()._init_kv_caches_from_forward_context(forward_context)
        self._refresh_hma_runtime_metadata()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.current_layer = 0
        self._wait_for_save_done = False

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, use register_kv_caches to init "
                "kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)
        else:
            self._refresh_hma_runtime_metadata()

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None
        self.layerwise_retrievers = []

        last_idx = None
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is not None and request.load_spec.can_load:
                last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            slot_mappings_by_group = _get_request_slot_mappings_on_device(self, request)
            _assert_non_layerwise_hma_path(self, slot_mappings_by_group, "start_load_kv")

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens

            if self.use_layerwise:
                slot_mapping = slot_mappings_by_group[0]
                assert len(tokens) == len(slot_mapping)
                sync = idx == last_idx

                if self.enable_blending:
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                retrieve_kwargs: dict[str, Any] = {
                    "kvcaches": kvcaches,
                    "slot_mappings_by_group": tuple(
                        slot_mapping[:lmcache_cached_tokens]
                        for slot_mapping in slot_mappings_by_group
                    ),
                    "block_ids_by_group": request.allocated_block_ids_by_group,
                    "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                    "request_configs": request.request_configs,
                    "req_id": request.req_id,
                }
                if len(slot_mappings_by_group) == 1:
                    retrieve_kwargs["slot_mapping"] = slot_mappings_by_group[0][
                        :lmcache_cached_tokens
                    ]

                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    **retrieve_kwargs,
                )

                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s retrieved %d tokens, expected %d tokens.",
                        request.req_id,
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(len(tokens))

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            if self.lmcache_engine is not None:
                for request in connector_metadata.requests:
                    self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            return

        if self.use_layerwise:
            assert not self.store_async, (
                "Layerwise storing is not supported with async store"
            )
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                self.lmcache_engine.lookup_unpin(request.req_id)
            self._wait_for_save_done = True
            self._replay_finished_stores_after_save()
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None
        self._refresh_hma_runtime_metadata()

        # lmcache-ascend start ---------------------
        ordering_event = torch.npu.Event()
        ordering_event.record()
        # lmcache-ascend end ---------------------

        for request in connector_metadata.requests:
            self.lmcache_engine.lookup_unpin(request.req_id)

            try:
                save_spec = request.save_spec
                if (
                    save_spec is None or not save_spec.can_save
                ) and self.kv_role != "kv_producer":
                    continue

                token_ids = request.token_ids
                raw_slot_mappings_by_group = request.slot_mappings_by_group
                if len(raw_slot_mappings_by_group) == 1:
                    slot_mappings_by_group = raw_slot_mappings_by_group
                else:
                    slot_mappings_by_group = tuple(
                        slot_mapping.to(self.device)
                        for slot_mapping in raw_slot_mappings_by_group
                    )
                _assert_non_layerwise_hma_path(
                    self,
                    slot_mappings_by_group,
                    "wait_for_save",
                )

                legacy_slot_mapping: Optional[torch.Tensor] = None
                slot_mapping_npu: Optional[torch.Tensor] = None
                if len(slot_mappings_by_group) == 1:
                    legacy_slot_mapping = slot_mappings_by_group[0]
                    assert len(legacy_slot_mapping) == len(token_ids)
                    # lmcache-ascend start ---------------------
                    pinned_slot_mapping = legacy_slot_mapping.pin_memory()
                    with torch.npu.stream(self.lmcache_engine.gpu_connector.store_stream):
                        slot_mapping_npu = pinned_slot_mapping.to(
                            device="npu",
                            dtype=torch.long,
                            non_blocking=True,
                        )
                    slot_mappings_by_group = (pinned_slot_mapping,)
                # lmcache-ascend end ---------------------

                skip_leading_tokens = save_spec.skip_leading_tokens

                if skip_leading_tokens == len(token_ids):
                    continue
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                is_last_prefill = request.is_last_prefill
                if is_last_prefill:
                    if request.disagg_spec:
                        request.disagg_spec.is_last_prefill = True
                elif not self.enable_blending:
                    aligned_token_len = (
                        len(token_ids)
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mappings_by_group = tuple(
                        slot_mapping[:aligned_token_len]
                        for slot_mapping in slot_mappings_by_group
                    )
                    if legacy_slot_mapping is not None:
                        legacy_slot_mapping = slot_mappings_by_group[0]

                store_kwargs: dict[str, Any] = {
                    "mask": store_mask,
                    "kvcaches": kvcaches,
                    "offset": skip_leading_tokens,
                    "transfer_spec": request.disagg_spec,
                    "request_configs": request.request_configs,
                    "req_id": request.req_id,
                    "ordering_event": ordering_event,
                }
                if len(slot_mappings_by_group) == 1:
                    assert legacy_slot_mapping is not None
                    store_kwargs["slot_mapping"] = legacy_slot_mapping
                    if slot_mapping_npu is not None:
                        store_kwargs["slot_mapping_npu"] = slot_mapping_npu
                else:
                    store_kwargs["slot_mappings_by_group"] = slot_mappings_by_group
                    store_kwargs["block_ids_by_group"] = (
                        request.allocated_block_ids_by_group
                    )

                self.lmcache_engine.store(token_ids, **store_kwargs)

                if get_pp_group().is_last_rank:
                    save_spec.skip_leading_tokens = len(token_ids)
                    if request.disagg_spec:
                        request.disagg_spec.num_transferred_tokens = len(token_ids)
            except Exception:
                logger.exception(
                    "wait_for_save failed for request %s; skipping save",
                    request.req_id,
                )
                continue

        self._wait_for_save_done = True
        self._replay_finished_stores_after_save()

    def _may_register_store_after_wait_for_save(self, request: "Request") -> bool:
        if self.kv_role == "kv_consumer":
            return False
        save_spec = request.save_spec
        if save_spec is None:
            return False
        if not save_spec.can_save and self.kv_role != "kv_producer":
            return False
        return save_spec.skip_leading_tokens != len(request.token_ids)

    def _replay_finished_stores_after_save(self) -> None:
        if not self._finished_req_ids_waiting_for_save or self.lmcache_engine is None:
            return

        finished_sending = self.lmcache_engine.get_finished_stores(
            self._finished_req_ids_waiting_for_save
        )
        if finished_sending:
            self._late_finished_sending |= finished_sending
        self._finished_req_ids_waiting_for_save = set()

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        if self.lmcache_engine is None:
            return None, None
        query_req_ids = set(finished_req_ids)
        if not self._wait_for_save_done:
            # NOTE (gingfung): The is a workaround logic for the case
            # where the requests is deferred (i.e. spec_decode or MTP)
            # and the model_runner call get_finished before wait_for_save.
            connector_metadata = self._parent._get_connector_metadata()
            assert isinstance(connector_metadata, LMCacheConnectorMetadata)

            waiting_for_save = {
                request.req_id
                for request in connector_metadata.requests
                if request.req_id in finished_req_ids
                and self._may_register_store_after_wait_for_save(request)
            }
            if waiting_for_save:
                self._finished_req_ids_waiting_for_save |= waiting_for_save
                query_req_ids -= waiting_for_save

        finished_sending = self.lmcache_engine.get_finished_stores(query_req_ids)
        if self._late_finished_sending:
            finished_sending |= self._late_finished_sending
            self._late_finished_sending = set()
        return (
            finished_sending if finished_sending else None,
            None,
        )

    def handle_preemptions(self, preempted_req_ids: set[str]) -> None:
        if self.lmcache_engine is None:
            return

        logger.debug(
            "LMCache-Ascend handling preemptions: req_ids=%s",
            sorted(preempted_req_ids),
        )

        # Lookup pins are request-scoped and normally released in wait_for_save().
        # A preempted request may leave that path before its metadata is replayed.
        for req_id in preempted_req_ids:
            self.lmcache_engine.lookup_unpin(req_id)

        if not self.store_async or self.kv_role == "kv_consumer":
            return

        waited_req_ids = self.lmcache_engine.wait_for_pending_stores(preempted_req_ids)
        if waited_req_ids:
            logger.info(
                "Handled preemptions after draining async stores: req_ids=%s",
                sorted(waited_req_ids),
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return self.request_finished_all_groups(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # The upstream 4.3 cleanup path is still keyed by request id; the
        # block ids are only part of the HMA connector interface here.
        primary_block_ids = block_ids[0] if block_ids else []
        _, return_params = super().request_finished(request, primary_block_ids)

        if (
            request.status == RequestStatus.FINISHED_ABORTED
            and self.lmcache_engine is not None
        ):
            self.lmcache_engine.lookup_unpin(request.request_id)

            if self.store_async and self.kv_role != "kv_consumer":
                try:
                    self.lmcache_engine.wait_for_pending_stores({request.request_id})
                except Exception:
                    logger.warning(
                        "wait_for_pending_stores failed for aborted request %s",
                        request.request_id,
                        exc_info=True,
                    )

        delay_free = self.store_async and self.kv_role != "kv_consumer"
        return delay_free, return_params
