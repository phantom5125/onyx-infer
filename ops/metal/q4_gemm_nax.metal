// Affine quantized matrix multiplication on the M5 neural accelerators.
//
// The kernel is MLX's affine_qmm_t_nax with its helpers (MIT, Apple Inc.),
// vendored under ops/metal/mlx_root. MLX dispatches it for FP16 and BF16
// activations when the device has neural accelerators. This file instantiates
// the BF16, group-64, 4-bit variants that Onyx dispatches.
//
// Bindings follow MLX so that the kernels stay interchangeable:
//   0 weights as uint32 words (8 nibbles, low nibble first)
//   1 scales, 2 affine biases, 3 activations, 4 output, all BF16
//   5 K, 6 N, 7 M as int32

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

#define instantiate_onyx_qmm_nax(aligned_n, bm, wm, wn)                              \
  instantiate_kernel(                                                                \
      "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_hgs_0_alN_" #aligned_n "_batch_0_bm_" #bm \
      "_bn_64_wm_" #wm "_wn_" #wn,                                                   \
      affine_qmm_t_nax,                                                              \
      bfloat16_t,                                                                    \
      64,                                                                            \
      4,                                                                             \
      false,                                                                         \
      aligned_n,                                                                     \
      false,                                                                         \
      bm,                                                                            \
      64,                                                                            \
      64,                                                                            \
      wm,                                                                            \
      wn)

// The staged 64x64 tile runs one flat threadgroup of four simdgroups.
instantiate_onyx_qmm_nax(1, 64, 4, 1);
// Threadgroup tiles of one or two 32-row blocks per simdgroup.
instantiate_onyx_qmm_nax(1, 32, 2, 2);
instantiate_onyx_qmm_nax(1, 64, 2, 2);
instantiate_onyx_qmm_nax(0, 32, 2, 2);
instantiate_onyx_qmm_nax(0, 64, 2, 2);
