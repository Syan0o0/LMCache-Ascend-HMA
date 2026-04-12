#include "mem_kernels.h"
#include "tiling/platform/platform_ascendc.h"
#include "utils.h"
#include <ATen/ATen.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>

namespace py = pybind11;

namespace {

void validate_gdn_memory_tensor(const torch::Tensor &memory_tensor,
                                const at::ScalarType expected_dtype,
                                const char *tensor_name) {
  TORCH_CHECK(memory_tensor.defined(),
              "GDN memory tensor for ", tensor_name, " must be defined.");
  TORCH_CHECK(memory_tensor.dim() >= 2,
              "GDN memory tensor for ", tensor_name,
              " must have at least 2 dims, got ", memory_tensor.dim(), ".");
  TORCH_CHECK(memory_tensor.scalar_type() == expected_dtype,
              "GDN memory tensor dtype mismatch for ", tensor_name,
              ". Expected ", expected_dtype, ", got ",
              memory_tensor.scalar_type(), ".");
  TORCH_CHECK(memory_tensor.is_contiguous(),
              "GDN memory tensor for ", tensor_name,
              " must be contiguous in phase 1.");
}

void validate_gdn_state_tensor(const torch::Tensor &state_tensor,
                               const at::ScalarType expected_dtype,
                               const torch::Tensor &memory_tensor,
                               const char *tensor_name, const int64_t layer_pos) {
  TORCH_CHECK(state_tensor.defined(), "GDN runtime tensor for ", tensor_name,
              " at layer ", layer_pos, " must be defined.");
  TORCH_CHECK(
      state_tensor.dim() == memory_tensor.dim() + 1,
      "GDN runtime tensor rank mismatch for ", tensor_name, " at layer ",
      layer_pos, ". Runtime rank should be transfer rank + 1, got runtime rank ",
      state_tensor.dim(), " and transfer rank ", memory_tensor.dim(), ".");
  TORCH_CHECK(state_tensor.scalar_type() == expected_dtype,
              "GDN runtime tensor dtype mismatch for ", tensor_name,
              " at layer ", layer_pos, ". Expected ", expected_dtype,
              ", got ", state_tensor.scalar_type(), ".");
  TORCH_CHECK(state_tensor.is_contiguous(), "GDN runtime tensor for ",
              tensor_name, " at layer ", layer_pos,
              " must be contiguous in phase 1.");

  for (int64_t dim = 1; dim < state_tensor.dim(); ++dim) {
    TORCH_CHECK(
        state_tensor.size(dim) == memory_tensor.size(dim - 1),
        "GDN runtime tensor tail shape mismatch for ", tensor_name,
        " at layer ", layer_pos, ". Runtime dim ", dim, " = ",
      state_tensor.size(dim), ", transfer dim ", dim - 1, " = ",
      memory_tensor.size(dim - 1), ".");
  }
}

int64_t compute_gdn_slice_numel(const torch::Tensor &memory_tensor,
                                const char *tensor_name) {
  TORCH_CHECK(memory_tensor.dim() >= 2,
              "GDN memory tensor for ", tensor_name,
              " must have at least 2 dims.");
  TORCH_CHECK(memory_tensor.size(0) > 0,
              "GDN memory tensor for ", tensor_name,
              " must have num_layers > 0.");
  TORCH_CHECK(memory_tensor.numel() % memory_tensor.size(0) == 0,
              "GDN memory tensor for ", tensor_name,
              " must be divisible by num_layers.");
  return memory_tensor.numel() / memory_tensor.size(0);
}

torch::Tensor build_gdn_state_ptr_tensor_on_device(
    const std::vector<torch::Tensor> &family_state_tensors,
    const torch::Device &runtime_device) {
  auto cpu_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto state_ptrs_cpu =
      torch::empty({static_cast<int64_t>(family_state_tensors.size())},
                   cpu_options);
  auto *state_ptrs_cpu_data = state_ptrs_cpu.data_ptr<int64_t>();

  for (size_t layer_pos = 0; layer_pos < family_state_tensors.size();
       ++layer_pos) {
    state_ptrs_cpu_data[layer_pos] =
        static_cast<int64_t>(family_state_tensors[layer_pos].data_ptr());
  }

  return state_ptrs_cpu.to(runtime_device);
}

void run_single_gdn_tensor_family_transfer(
    torch::Tensor &memory_tensor,
    const std::vector<torch::Tensor> &family_state_tensors,
    const int64_t block_id, const bool direction, const char *tensor_name) {
  TORCH_CHECK(!family_state_tensors.empty(),
              "GDN runtime tensor family ", tensor_name, " must not be empty.");

  const auto runtime_device = family_state_tensors[0].device();
  const auto num_layers = static_cast<int32_t>(family_state_tensors.size());
  const auto slice_numel = compute_gdn_slice_numel(memory_tensor, tensor_name);

  auto state_ptrs_on_device =
      build_gdn_state_ptr_tensor_on_device(family_state_tensors, runtime_device);
  auto config = prepare_gdn_state_transfer_config(
      memory_tensor, runtime_device, num_layers, slice_numel, direction);

  uint8_t *memory_tensor_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(memory_tensor);
  uint8_t *state_ptrs_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(state_ptrs_on_device);

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_gdn_state_transfer_kernel");
  cmd.SetCustomHandler([config, memory_tensor_ptr, state_ptrs_ptr, block_id,
                        state_ptrs_on_device]() -> int {
    (void)state_ptrs_on_device;
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
    kvcache_ops::multi_layer_gdn_state_transfer_kernel(
        dtype_num, config.aiv_num, config.stream, memory_tensor_ptr,
        state_ptrs_ptr, block_id, config.num_layers, config.slice_numel,
        config.direction);
    return 0;
  });
  cmd.Run();
}

} // namespace

/**
 * Quickly offload KV cache from vLLM paged memory to the offloading buffer
 * Processes all the layers at the same time
 *
 * Each layer in vLLM's KV buffer has a shape of
 * [2, PAGE_BUFFER_SIZE, num_heads*head_size]
 *
 * Each AIV Core processes the copy for a token
 *
 * Therefore:
 *  AIV Core - token
 *
 * The function does:
 * slot_id = slot_mapping[tokenId]
 * ptrs[mem_offset(kv, layer, tokenId, hiddenDims)] = key_value[mem_offset(kv,
 * layer, pages, pageSize, slot_id, hiddenDims)]
 *
 * Param:
 *  - direction: false  means LMCache to PagedBuffer, true  means PagedBuffer to
 * LMCache
 */
void multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int kvcache_format_raw) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  // Calculate UB buffer parameters
  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, key_value_ptr]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, key_value_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction);
    return 0;
  });
  cmd.Run();
  return;
};

void fused_multi_layer_kv_transfer(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    torch::Tensor &staging_cache,        // staging buffer
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, // true: from_gpu, false: to_gpu
    const bool use_mla, const int kvcache_format_raw) {
  // get host cpu buffer pointer for aclrtMemcpyAsync
  uint8_t *key_value_ptr = static_cast<uint8_t *>(key_value.data_ptr());
  uint8_t *staging_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(staging_cache);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  compute_multi_layer_ub_params(config, key_value, paged_memory_device,
                                key_value_ptrs);

  // Calculate and verify the CPU buffer size
  size_t cpu_buffer_size = static_cast<size_t>(config.kv_size) *
                           config.num_layers * config.num_tokens *
                           config.hidden_dims * key_value.element_size();

  TORCH_CHECK(
      staging_cache.numel() * staging_cache.element_size() >= cpu_buffer_size,
      "staging_cache size insufficient: need ", cpu_buffer_size, " bytes, got ",
      staging_cache.numel() * staging_cache.element_size());

  at_npu::native::OpCommand cmd;
  cmd.Name("fused_multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([config, staging_cache_ptr, key_value_ptr,
                        cpu_buffer_size]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);

    aclError ret;
    // direction: false = to_gpu (H2D), true = from_gpu (D2H)
    bool isH2D = !config.direction;

    // Step 1: H2D memcpy (to_gpu) currently not used
    if (isH2D) {
      ret = aclrtMemcpyAsync(staging_cache_ptr, cpu_buffer_size, key_value_ptr,
                             cpu_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "H2D memcpy failed: cpu_buffer -> staging_cache, ret=", ret);
    }

    // Step 2: Kernel (Gather or Scatter)
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, config.kvcache_format, config.aiv_num,
        config.stream, config.page_buffer_ptrs, staging_cache_ptr,
        config.slot_mapping_ptr, config.hidden_dims, config.kv_size,
        config.num_layers, config.page_buffer_size, config.num_tokens,
        config.singlePerLoopBuffer, config.maxTokensPerLoop, config.direction);

    // Step 3: D2H memcpy (from_gpu)
    if (!isH2D) {
      ret = aclrtMemcpyAsync(key_value_ptr, cpu_buffer_size, staging_cache_ptr,
                             cpu_buffer_size, ACL_MEMCPY_DEVICE_TO_HOST,
                             config.stream);
      TORCH_CHECK(ret == ACL_ERROR_NONE,
                  "D2H memcpy failed: staging_cache -> cpu_buffer, ret=", ret);
    }

    return 0;
  });
  cmd.Run();
  return;
}

void multi_layer_gdn_state_transfer(
    std::vector<torch::Tensor> &memory_tensors,
    std::vector<torch::Tensor> &state_tensors, const int64_t block_id,
    const bool direction) {
  TORCH_CHECK(memory_tensors.size() == 2,
              "First-version multi_layer_gdn_state_transfer expects exactly 2 "
              "memory tensors, got ",
              memory_tensors.size(), ".");
  TORCH_CHECK(!state_tensors.empty(),
              "multi_layer_gdn_state_transfer requires non-empty state_tensors.");

  auto &conv_memory_tensor = memory_tensors[0];
  auto &ssm_memory_tensor = memory_tensors[1];
  validate_gdn_memory_tensor(conv_memory_tensor, at::kBFloat16, "conv_state");
  validate_gdn_memory_tensor(ssm_memory_tensor, at::kFloat, "ssm_state");

  const int64_t num_layers = conv_memory_tensor.size(0);
  TORCH_CHECK(ssm_memory_tensor.size(0) == num_layers,
              "GDN memory tensors must agree on num_layers. conv_state has ",
              num_layers, ", ssm_state has ", ssm_memory_tensor.size(0), ".");
  TORCH_CHECK(num_layers > 0,
              "multi_layer_gdn_state_transfer requires num_layers > 0.");
  TORCH_CHECK(state_tensors.size() == static_cast<size_t>(num_layers * 2),
              "First-version multi_layer_gdn_state_transfer expects ",
              num_layers * 2, " runtime state tensors, got ",
              state_tensors.size(), ".");
  TORCH_CHECK(block_id >= 0,
              "multi_layer_gdn_state_transfer requires block_id >= 0, got ",
              block_id, ".");
  const auto runtime_device = state_tensors[0].device();

  for (int64_t layer_pos = 0; layer_pos < num_layers; ++layer_pos) {
    auto &conv_state_tensor = state_tensors[layer_pos];
    auto &ssm_state_tensor = state_tensors[num_layers + layer_pos];

    TORCH_CHECK(conv_state_tensor.device() == runtime_device,
                "All conv_state runtime tensors must live on the same device.");
    TORCH_CHECK(ssm_state_tensor.device() == runtime_device,
                "All ssm_state runtime tensors must live on the same device.");

    validate_gdn_state_tensor(conv_state_tensor, at::kBFloat16,
                              conv_memory_tensor, "conv_state", layer_pos);
    validate_gdn_state_tensor(ssm_state_tensor, at::kFloat, ssm_memory_tensor,
                              "ssm_state", layer_pos);

    TORCH_CHECK(block_id < conv_state_tensor.size(0),
                "block_id ", block_id,
                " is out of range for conv_state runtime tensor at layer ",
                layer_pos, " with block dim ", conv_state_tensor.size(0), ".");
    TORCH_CHECK(block_id < ssm_state_tensor.size(0),
                "block_id ", block_id,
                " is out of range for ssm_state runtime tensor at layer ",
                layer_pos, " with block dim ", ssm_state_tensor.size(0), ".");
  }

  std::vector<torch::Tensor> conv_state_tensors;
  std::vector<torch::Tensor> ssm_state_tensors;
  conv_state_tensors.reserve(num_layers);
  ssm_state_tensors.reserve(num_layers);
  for (int64_t layer_pos = 0; layer_pos < num_layers; ++layer_pos) {
    conv_state_tensors.push_back(state_tensors[layer_pos]);
    ssm_state_tensors.push_back(state_tensors[num_layers + layer_pos]);
  }

  run_single_gdn_tensor_family_transfer(conv_memory_tensor, conv_state_tensors,
                                        block_id, direction, "conv_state");
  run_single_gdn_tensor_family_transfer(ssm_memory_tensor, ssm_state_tensors,
                                        block_id, direction, "ssm_state");
}

void multi_layer_kv_transfer_310p(
    torch::Tensor &key_value,            // [kv, num_layer, num_tokens, hidden]
    const torch::Tensor &key_value_ptrs, // [num_layers]
    const torch::Tensor &slot_mapping,   // [num_tokens]
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction, const bool use_mla, const int num_kv_head,
    const int head_size, const int blockSize, const int kvcache_format_raw) {
  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);

  MultiLayerKVConfig config = prepare_multi_layer_kv_config(
      key_value, key_value_ptrs, slot_mapping, paged_memory_device,
      page_buffer_size, direction, use_mla, kvcache_format_raw);

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);
  // we require the kv ptr list to be on the device too
  const c10::OptionalDeviceGuard kv_device_guard(device_of(key_value_ptrs));

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_310p");
  cmd.SetCustomHandler([config, stream, key_value_ptr, num_kv_head, head_size,
                        blockSize]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(config.slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(config.scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(config.socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::multi_layer_kv_transfer_kernel_310p(
        dtype_num, slot_num, config.kvcache_format, aiv_num, stream,
        config.page_buffer_ptrs, key_value_ptr, config.slot_mapping_ptr,
        config.hidden_dims, config.kv_size, config.num_layers,
        config.page_buffer_size, config.num_tokens, config.direction,
        num_kv_head, head_size, blockSize);
    return 0;
  });
  cmd.Run();
  return;
};

void multi_layer_kv_transfer_unilateral(
    torch::Tensor &key_value, const torch::Tensor &key_ptrs,
    const torch::Tensor &value_ptrs, const torch::Tensor &slot_mapping,
    const torch::Device &paged_memory_device, const int page_buffer_size,
    const bool direction) {
  // TODO:
  PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
  throw py::error_already_set();
};

void single_layer_kv_transfer(
    torch::Tensor
        &lmc_key_value_cache, // [num_tokens, 2, num_heads*head_size]
                              // or [2, num_tokens, num_heads*head_size]
    std::vector<torch::Tensor> &vllm_kv_caches,
    // SEPARATE_KV: list[k_tensor, v_tensor]
    // k_tensor/v_tensor = [num_blocks, block_size, num_heads, head_size]
    // MERGED_KV:
    // vllm_two_major=true:  [2, num_blocks, block_size, num_heads, head_size]
    // vllm_two_major=false: [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens]
    const bool direction, // false: LMCache -> Paged, true: Paged -> LMCache
    const int kvcache_format_raw, // 1: MERGED_KV, 2: SEPARATE_KV
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major // true: [2, blocks, ...], false: [blocks, 2, ...]
                              // (only for MERGED_KV)
) {
  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(device_of(slot_mapping));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      lmc_key_value_cache, vllm_kv_caches, slot_mapping, direction, token_major,
      vllm_two_major, is_separate);

  at_npu::native::OpCommand cmd;
  cmd.Name(is_separate ? "single_layer_kv_transfer_kernel_v2_separate"
                       : "single_layer_kv_transfer_kernel_v2");

  cmd.SetCustomHandler([config, is_separate]() -> int {
    if (!is_separate) {
      // Merged KV Kernel
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, config.direction, config.token_major);
    } else {
      // Separate KV Kernel
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          config.direction, config.token_major);
    }
    return 0;
  });
  cmd.Run();
}

void batched_fused_single_layer_kv_transfer(
    std::vector<torch::Tensor>
        &lmc_tensors, // N CPU pinned memory tensors
                      // token_major=true:  [num_tokens, 2, num_heads*head_size]
                      // token_major=false: [2, num_tokens, num_heads*head_size]
    torch::Tensor &staging_cache, // NPU staging buffer
                                  // token_major=true:  [num_tokens, 2,
                                  // num_heads*head_size] token_major=false: [2,
                                  // num_tokens, num_heads*head_size]
    std::vector<torch::Tensor>    // separate format： list[k_tensor, v_tensor]
        &vllm_kv_caches, // k_tensor/v_tensor = [num_blocks，block_size,
                         // num_heads, head_size]
                         //  Mergeed format：
                         //  vllm_two_major=true:  [2, num_blocks, block_size,
                         //  num_heads, head_size] vllm_two_major=false:
                         //  [num_blocks, 2, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping_full, // [num_tokens]
    std::vector<int64_t>
        &chunk_offsets,                // token offset in staging for each chunk
    std::vector<int64_t> &chunk_sizes, // token count for each chunk
    const bool direction, // false: CPU -> staging -> paged (to_gpu) true: paged
                          // -> staging -> CPU (from_gpu)
    const int kvcache_format_raw,
    const bool
        token_major, // true: [tokens, 2, hidden], false: [2, tokens, hidden]
    const bool vllm_two_major // true: [2, blocks, ...], false: [blocks, 2, ...]
) {

  bool is_separate = validate_vllm_caches(vllm_kv_caches, kvcache_format_raw);

  const c10::OptionalDeviceGuard slot_device_guard(
      device_of(slot_mapping_full));

  SingleLayerKVConfig config = prepare_single_layer_kv_config(
      staging_cache, vllm_kv_caches, slot_mapping_full, direction, token_major,
      vllm_two_major, is_separate);

  int64_t element_size = staging_cache.element_size();

  if (!is_separate) {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_val_offset, config.strides.vllm_k_bytes,
          config.strides.lmc_token_stride, config.strides.lmc_val_offset,
          config.strides.lmc_bytes, config.ub_params.max_tokens_per_loop,
          config.dims.num_heads, config.dims.head_dims, config.dims.num_tokens,
          config.dims.block_size, is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);

  } else {
    auto launcher = [config](bool is_gather) {
      kvcache_ops::single_layer_kv_transfer_kernel_v2_separate(
          config.ub_params.scalar_type_num, config.ub_params.slot_type_num,
          config.ub_params.aiv_num, config.ub_params.stream,
          config.ptrs.lmc_ptr, config.ptrs.vllm_k_ptr, config.ptrs.vllm_v_ptr,
          config.ptrs.slot_mapping_ptr, config.strides.vllm_k_stride,
          config.strides.vllm_v_stride, config.strides.vllm_k_bytes,
          config.strides.vllm_v_bytes, config.strides.lmc_token_stride,
          config.strides.lmc_val_offset, config.strides.lmc_bytes,
          config.ub_params.max_tokens_per_loop, config.dims.num_heads,
          config.dims.head_dims, config.dims.num_tokens, config.dims.block_size,
          is_gather, config.token_major);
    };
    run_batched_fused_transfer(config, lmc_tensors, chunk_offsets, chunk_sizes,
                               element_size, launcher);
  }
}

void load_and_reshape_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();
  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("load_and_reshape_flash_kernel");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, true);
    return 0;
  });
  cmd.Run();
  return;
};

void reshape_and_cache_back_flash(
    torch::Tensor &key_value, // [2, num_layer, num_tokens, num_heads*head_size]
                              // must be one gpu / pinned cpu
    torch::Tensor &key_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor
        &value_cache, // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor &slot_mapping, // [num_tokens],
    const int layer_idx) {

  uint8_t *key_value_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_value);
  uint8_t *key_cache_ptr = get_kernel_ptr<uint8_t, torch::Tensor>(key_cache);
  uint8_t *value_cache_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(value_cache);

  uint8_t *slot_mapping_ptr =
      get_kernel_ptr<uint8_t, torch::Tensor>(slot_mapping);

  int num_tokens = slot_mapping.size(0);
  int num_layers = key_value.size(1);
  int block_size = key_cache.size(1);
  int num_blocks = key_cache.size(0);
  int hidden_dims = key_value.size(-1);
  const c10::OptionalDeviceGuard device_guard(device_of(key_cache));
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();

  const char *socName = aclrtGetSocName();

  at_npu::native::OpCommand cmd;
  cmd.Name("reshape_and_cache_back_flash");
  cmd.SetCustomHandler([scalar_type, slot_type, socName, stream, key_value_ptr,
                        key_cache_ptr, value_cache_ptr, slot_mapping_ptr,
                        hidden_dims, num_blocks, block_size, num_tokens,
                        num_layers, layer_idx]() -> int {
    auto slot_num = vllm_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = vllm_ascend::get_dtype_from_torch(scalar_type);
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    kvcache_ops::load_and_reshape_flash_kernel(
        dtype_num, slot_num, aiv_num, stream, key_value_ptr, key_cache_ptr,
        value_cache_ptr, slot_mapping_ptr, hidden_dims, num_blocks, block_size,
        num_tokens, num_layers, layer_idx, false);
    return 0;
  });
  cmd.Run();
  return;
};
