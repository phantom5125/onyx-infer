// Onyx Q4 GEMM — simdgroup MMA variant.
//
// Derived from Apple MLX Steel GEMM (MIT License, Copyright © 2023–2024 Apple Inc.).
// See MLX quantized.h and the Steel GEMM mma.h and loader.h headers.
// The design stages tiles in threadgroup memory and dequantizes weights per thread.
// This variant specializes the layout for Onyx Q4 with BM=BN=BK=32 and WM=WN=2.
// It uses float scales and an implicit zero point of 8.
//
// Observations from the M5 (AGXG17C) experiments on November 18, 2026.
// Metal 4 simdgroup_load/store use the row stride in elements.
// Interleaving operand loads with simdgroup stores corrupted accumulators in the probe.
// Use scalar reads and thread_elements reinterpretation as in MLX BaseMMAFrag.
//
// Dispatch contract:
//   - k % 32 == 0、group_size % 32 == 0、n % 32 == 0
// The host may pad M to 32-row tiles. The kernel also guards the M edge.
//
// Input layout:
//   activation [m][k]        half, row-major
//   packed     [n][k/2]      uchar, low nibble at even k, decoded as (q - 8) * scale
//   scales      [n][k/group]  float
//   output      [m][n]        half
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Metal-native BF16 activation type with the FP32 exponent range.
typedef bfloat bfloat16_t;

constant constexpr short BM = 32;
constant constexpr short BN = 32;
constant constexpr short BK = 32;
constant constexpr short LD_TGP = 34;  // Padded threadgroup row stride.
constant constexpr short WN = 2;
constant constexpr short FRAG_STRIDE_A = 16 * LD_TGP;  // Spacing between row fragments of A.
constant constexpr short FRAG_STRIDE_B = 16;               // Spacing between column fragments of B.

// Map a lane to coordinates in an 8×8 fragment, following MLX get_coord.
// thread_elements() holds two adjacent columns at the same row.
METAL_FUNC short2 frag_coord(ushort simd_lid) {
  const short qid = simd_lid / 4;
  const short fm = (qid & 4) + ((simd_lid / 2) % 4);
  const short fn = (qid & 2) * 2 + (simd_lid % 2) * 2;
  return short2{fn, fm};
}

// Decode one packed byte into two threadgroup half values with the same scale.
// row_stride advances to the next K row. The low nibble is written first.
METAL_FUNC void onyx_dequant2(uchar byte, float scale, threadgroup half* dst, uint row_stride) {
  const float s = scale;
  dst[0] = half(s * float(byte & 0x0f) - 8.0f * s);
  dst[row_stride] = half(s * float(byte >> 4) - 8.0f * s);
}

struct alignas(16) AVec8 {
  uchar v[16];  // Eight half values.
};


// Decode in float and round to native BF16.
METAL_FUNC void onyx_dequant2_bf16(uchar byte, float scale, threadgroup bfloat16_t* dst, uint row_stride) {
  const float s = scale;
  dst[0] = bfloat16_t(s * float(byte & 0x0f) - 8.0f * s);
  dst[row_stride] = bfloat16_t(s * float(byte >> 4) - 8.0f * s);
}

kernel void onyx_q4_gemm_simd_bn32_bk32_bf16(
    device const bfloat16_t* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    device const ushort* bias_bits [[buffer(8)]],
    uint2 tid [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]]) {
  if ((k % 32) || (group_size % 32) || (n % 32)) return;  // Reject a shape that violates the dispatch contract.

  threadgroup bfloat16_t Xs[BM * LD_TGP];   // A tile [m][k]
  threadgroup bfloat16_t Ws[BN * LD_TGP];   // Stage transposed weights for MMA.
  threadgroup float Qs[BM * LD_TGP];  // Stage the accumulator tile.

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const int lane = simd_gid * 32 + simd_lid;
  const int groups_per_row = k / int(group_size);
  const int group_steps = int(group_size) / BK;  // Advance to the next scale after this many K tiles.

  // Load eight activation elements per thread.
  const int a_row = lane / 4;
  const int a_col = 8 * (lane % 4);
  const device bfloat16_t* a_src = activation + (uint)(y_row + a_row) * k + a_col;
  threadgroup bfloat16_t* a_dst = Xs + a_row * LD_TGP + a_col;

  // Decode four packed bytes per lane into the [k][n] weight tile.
  const int w_row = lane / 4;                      // Column within the weight tile.
  const int w_kbase = 8 * (lane % 4);              // Base K coordinate within the tile.
  const device uchar* w_src = packed_weights + (uint)(y_col + w_row) * (k / 2) + w_kbase / 2;
  const device float* w_scale = scales + (uint)(y_col + w_row) * groups_per_row;

  // Four simdgroups cover the four 16×16 quadrants of a 32×32 tile.
  const int tm = 8 * (simd_gid / WN);  // M quadrant offset.
  const int tn = 8 * (simd_gid % WN);  // N quadrant offset.
  const short2 lc = frag_coord(simd_lid);
  const int sm = tm + lc.y;  // A fragment row for this lane.
  const int sn = tn + lc.x;  // B fragment column for this lane.

  simdgroup_matrix<float, 8, 8> acc00, acc01, acc10, acc11;
  acc00 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc01 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc10 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc11 = simdgroup_matrix<float, 8, 8>(0.0f);

  const int k_tiles = k / BK;
  int group_step_cnt = 0;
  for (int kk = 0; kk < k_tiles; ++kk) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {  // Copy eight activation elements and zero-fill rows beyond M.
      if ((uint)(y_row + a_row) < m) {
        for (int i = 0; i < 8; ++i) a_dst[i] = a_src[i];
      } else {
        for (int i = 0; i < 8; ++i) a_dst[i] = bfloat16_t(0);
      }
    }
    {  // Decode eight weights into transposed threadgroup storage.
      const float s = *w_scale;
      threadgroup bfloat16_t* dst0 = Ws + w_kbase * LD_TGP + w_row;
      for (int i = 0; i < 4; ++i) {
        onyx_dequant2_bf16(w_src[i], s, dst0 + i * 2 * LD_TGP, LD_TGP);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read the two elements owned by each lane using the MLX fragment layout.
    // Use thread_elements to populate each MMA fragment.
    // Accumulate from zero to keep the loop branch-free and unrollable.
    // A fragment i: row = sm + 16i, column = k8 + lc.x + {0,1}.
    // B fragment j: row = k8 + lc.y, column = sn + 16j + lc.x + {0,1}.
    for (short k8 = 0; k8 < BK; k8 += 8) {
      const threadgroup bfloat16_t* a_ptr = Xs + sm * LD_TGP + k8 + lc.x;
      const threadgroup bfloat16_t* b_ptr = Ws + (k8 + lc.y) * LD_TGP + sn;
      simdgroup_matrix<float, 8, 8> a_mat0, a_mat1, b_mat0, b_mat1;
      reinterpret_cast<thread vec<float, 2>&>(a_mat0.thread_elements()) =
          vec<float, 2>(float(a_ptr[0]), float(a_ptr[1]));
      reinterpret_cast<thread vec<float, 2>&>(a_mat1.thread_elements()) =
          vec<float, 2>(float(a_ptr[FRAG_STRIDE_A]), float(a_ptr[FRAG_STRIDE_A + 1]));
      reinterpret_cast<thread vec<float, 2>&>(b_mat0.thread_elements()) =
          vec<float, 2>(float(b_ptr[0]), float(b_ptr[1]));
      reinterpret_cast<thread vec<float, 2>&>(b_mat1.thread_elements()) =
          vec<float, 2>(float(b_ptr[FRAG_STRIDE_B]), float(b_ptr[FRAG_STRIDE_B + 1]));
      simdgroup_multiply_accumulate(acc00, a_mat0, b_mat0, acc00);
      simdgroup_multiply_accumulate(acc01, a_mat0, b_mat1, acc01);
      simdgroup_multiply_accumulate(acc10, a_mat1, b_mat0, acc10);
      simdgroup_multiply_accumulate(acc11, a_mat1, b_mat1, acc11);
    }

    // Advance device offsets and the scale index.
    a_src += BK;
    w_src += BK / 2;
    group_step_cnt++;
    if (group_step_cnt == group_steps) {
      group_step_cnt = 0;
      w_scale++;
    }
  }

  // Each simdgroup cooperatively stores its four fragments into Qs.
  // After the barrier, all 128 lanes write linear output elements.
  simdgroup_store(acc00, Qs + tm * LD_TGP + tn, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc01, Qs + tm * LD_TGP + tn + 16, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc10, Qs + (tm + 16) * LD_TGP + tn, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc11, Qs + (tm + 16) * LD_TGP + tn + 16, LD_TGP, ulong2(0, 0), false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    // The 128 lanes write eight elements each for a 32×32 output tile.
    const int o_row = lane / 4;
    const int o_col = 8 * (lane % 4);
    if ((uint)(y_row + o_row) < m) {  // Skip rows beyond M.
      for (int c = 0; c < 8; ++c) {
        const uint32_t bias_f32 = uint32_t(bias_bits[y_col + o_col + c]) << 16;
        const float bias_value = as_type<float>(bias_f32);  // Expand BF16 bits to FP32.
        output[(uint)(y_row + o_row) * n + (y_col + o_col + c)] =
            bfloat16_t(Qs[o_row * LD_TGP + o_col + c] + bias_value);
      }
    }
  }
}

kernel void onyx_q4_gemm_simd_bn32_bk32(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    device const ushort* bias_bits [[buffer(8)]],
    uint2 tid [[threadgroup_position_in_grid]],
    ushort simd_gid [[simdgroup_index_in_threadgroup]],
    ushort simd_lid [[thread_index_in_simdgroup]]) {
  if ((k % 32) || (group_size % 32) || (n % 32)) return;  // Reject a shape that violates the dispatch contract.

  threadgroup half Xs[BM * LD_TGP];   // A tile [m][k]
  threadgroup half Ws[BN * LD_TGP];   // Stage transposed weights for MMA.
  threadgroup float Qs[BM * LD_TGP];  // Stage the accumulator tile.

  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const int lane = simd_gid * 32 + simd_lid;
  const int groups_per_row = k / int(group_size);
  const int group_steps = int(group_size) / BK;  // Advance to the next scale after this many K tiles.

  // Load eight activation elements per thread.
  const int a_row = lane / 4;
  const int a_col = 8 * (lane % 4);
  const device half* a_src = activation + (uint)(y_row + a_row) * k + a_col;
  threadgroup half* a_dst = Xs + a_row * LD_TGP + a_col;

  // Decode four packed bytes per lane into the [k][n] weight tile.
  const int w_row = lane / 4;                      // Column within the weight tile.
  const int w_kbase = 8 * (lane % 4);              // Base K coordinate within the tile.
  const device uchar* w_src = packed_weights + (uint)(y_col + w_row) * (k / 2) + w_kbase / 2;
  const device float* w_scale = scales + (uint)(y_col + w_row) * groups_per_row;

  // Four simdgroups cover the four 16×16 quadrants of a 32×32 tile.
  const int tm = 8 * (simd_gid / WN);  // M quadrant offset.
  const int tn = 8 * (simd_gid % WN);  // N quadrant offset.
  const short2 lc = frag_coord(simd_lid);
  const int sm = tm + lc.y;  // A fragment row for this lane.
  const int sn = tn + lc.x;  // B fragment column for this lane.

  simdgroup_matrix<float, 8, 8> acc00, acc01, acc10, acc11;
  acc00 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc01 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc10 = simdgroup_matrix<float, 8, 8>(0.0f);
  acc11 = simdgroup_matrix<float, 8, 8>(0.0f);

  const int k_tiles = k / BK;
  int group_step_cnt = 0;
  for (int kk = 0; kk < k_tiles; ++kk) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {  // Copy eight activation elements and zero-fill rows beyond M.
      if ((uint)(y_row + a_row) < m) {
        for (int i = 0; i < 8; ++i) a_dst[i] = a_src[i];
      } else {
        for (int i = 0; i < 8; ++i) a_dst[i] = half(0);
      }
    }
    {  // Decode eight weights into transposed threadgroup storage.
      const float s = *w_scale;
      threadgroup half* dst0 = Ws + w_kbase * LD_TGP + w_row;
      for (int i = 0; i < 4; ++i) {
        onyx_dequant2(w_src[i], s, dst0 + i * 2 * LD_TGP, LD_TGP);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Read the two elements owned by each lane using the MLX fragment layout.
    // Use thread_elements to populate each MMA fragment.
    // Accumulate from zero to keep the loop branch-free and unrollable.
    // A fragment i: row = sm + 16i, column = k8 + lc.x + {0,1}.
    // B fragment j: row = k8 + lc.y, column = sn + 16j + lc.x + {0,1}.
#pragma unroll
    for (short k8 = 0; k8 < BK; k8 += 8) {
      const threadgroup half* a_ptr = Xs + sm * LD_TGP + k8 + lc.x;
      const threadgroup half* b_ptr = Ws + (k8 + lc.y) * LD_TGP + sn;
      simdgroup_matrix<float, 8, 8> a_mat0, a_mat1, b_mat0, b_mat1;
      reinterpret_cast<thread vec<float, 2>&>(a_mat0.thread_elements()) =
          vec<float, 2>(float(a_ptr[0]), float(a_ptr[1]));
      reinterpret_cast<thread vec<float, 2>&>(a_mat1.thread_elements()) =
          vec<float, 2>(float(a_ptr[FRAG_STRIDE_A]), float(a_ptr[FRAG_STRIDE_A + 1]));
      reinterpret_cast<thread vec<float, 2>&>(b_mat0.thread_elements()) =
          vec<float, 2>(float(b_ptr[0]), float(b_ptr[1]));
      reinterpret_cast<thread vec<float, 2>&>(b_mat1.thread_elements()) =
          vec<float, 2>(float(b_ptr[FRAG_STRIDE_B]), float(b_ptr[FRAG_STRIDE_B + 1]));
      simdgroup_multiply_accumulate(acc00, a_mat0, b_mat0, acc00);
      simdgroup_multiply_accumulate(acc01, a_mat0, b_mat1, acc01);
      simdgroup_multiply_accumulate(acc10, a_mat1, b_mat0, acc10);
      simdgroup_multiply_accumulate(acc11, a_mat1, b_mat1, acc11);
    }

    // Advance device offsets and the scale index.
    a_src += BK;
    w_src += BK / 2;
    group_step_cnt++;
    if (group_step_cnt == group_steps) {
      group_step_cnt = 0;
      w_scale++;
    }
  }

  // Each simdgroup cooperatively stores its four fragments into Qs.
  // After the barrier, all 128 lanes write linear output elements.
  simdgroup_store(acc00, Qs + tm * LD_TGP + tn, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc01, Qs + tm * LD_TGP + tn + 16, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc10, Qs + (tm + 16) * LD_TGP + tn, LD_TGP, ulong2(0, 0), false);
  simdgroup_store(acc11, Qs + (tm + 16) * LD_TGP + tn + 16, LD_TGP, ulong2(0, 0), false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    // The 128 lanes write eight elements each for a 32×32 output tile.
    const int o_row = lane / 4;
    const int o_col = 8 * (lane % 4);
    if ((uint)(y_row + o_row) < m) {  // Skip rows beyond M.
      for (int c = 0; c < 8; ++c) {
        const uint32_t bias_f32 = uint32_t(bias_bits[y_col + o_col + c]) << 16;
        const float bias_value = as_type<float>(bias_f32);  // Expand BF16 bits to FP32.
        output[(uint)(y_row + o_row) * n + (y_col + o_col + c)] =
            half(Qs[o_row * LD_TGP + o_col + c] + bias_value);
      }
    }
  }
}
