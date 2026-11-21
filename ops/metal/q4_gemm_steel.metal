// Onyx Q4 GEMM, based on MLX Steel QMM.
//
// Instantiate the MLX affine_qmm_t kernel with group size 64 and four-bit weights.
// Use unbatched 32×32×32 tiles for the native runtime and operator probes.
//
// Apple MLX, MIT License, Copyright © 2023–2024 Apple Inc.
// The vendored headers and license are under ops/metal/mlx_root/.
// Onyx decodes (nibble - 8) * scale. MLX computes nibble * scale + bias.
// The host supplies bias = -8 * scale.
// Both expressions implement the same affine transform.
//
// Packed weights are [n][k/2] bytes, with even K values in the low nibble.
// Instantiate float16_t and bfloat16_t variants.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define instantiate_qmm(name, type, group_size, bits, aligned_n, batched) \
  instantiate_kernel(                                                     \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_hgs_0_alN_" #aligned_n "_batch_" #batched "_bm_32_bn_32", \
      name,                                                               \
      type,                                                               \
      group_size,                                                         \
      bits,                                                               \
      false,                                                              \
      aligned_n,                                                          \
      batched,                                                            \
      32,                                                                 \
      32,                                                                 \
      32)

instantiate_qmm(affine_qmm_t, float16_t, 64, 4, 0, 0)
instantiate_qmm(affine_qmm_t, bfloat16_t, 64, 4, 0, 0)

// Convert BF16 input to FP16 on the GPU.
// Expand BF16 bits to FP32 before converting to FP16.
kernel void onyx_bf16_to_fp16(
    device const uint16_t* input [[buffer(0)]],
    device uint16_t* output [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  uint32_t f32_bits = uint32_t(input[id]) << 16;
  auto f = as_type<float>(f32_bits);
  output[id] = as_type<uint16_t>(half(f));
}

// Convert FP16 output to BF16 on the GPU.
// Round to nearest even using the retained low bit.
kernel void onyx_fp16_to_bf16(
    device const uint16_t* input [[buffer(0)]],
    device uint16_t* output [[buffer(1)]],
    constant uint32_t& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= count) return;
  // Decode the FP16 value through FP32 before rounding to BF16.
  // Use the same round-to-nearest-even rule as float_to_bf16.
  const half h = as_type<half>(input[id]);
  const float f = static_cast<float>(h);
  const uint32_t f32_bits = as_type<uint32_t>(f);
  const uint32_t rounding = 0x7fffu + ((f32_bits >> 16) & 1u);
  output[id] = uint16_t((f32_bits + rounding) >> 16);
}
