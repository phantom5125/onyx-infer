#include <metal_stdlib>

using namespace metal;

constant constexpr uint TILE = 16;

kernel void onyx_q4_gemm_fp16acc32(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 tile_position [[threadgroup_position_in_grid]],
    uint2 local_position [[thread_position_in_threadgroup]]) {
  threadgroup half activation_tile[TILE][TILE];
  threadgroup float weight_tile[TILE][TILE];
  const uint row = tile_position.y * TILE + local_position.y;
  const uint column = tile_position.x * TILE + local_position.x;
  float sum = 0.0f;
  const uint groups = k / group_size;

  for (uint tile_k = 0; tile_k < k; tile_k += TILE) {
    const uint input_k = tile_k + local_position.x;
    activation_tile[local_position.y][local_position.x] =
        row < m && input_k < k ? activation[row * k + input_k] : half(0.0h);

    const uint weight_row = tile_position.x * TILE + local_position.y;
    const uint weight_k = tile_k + local_position.x;
    float weight = 0.0f;
    if (weight_row < n && weight_k < k) {
      const uint byte_index = weight_row * (k / 2) + weight_k / 2;
      const uint packed = packed_weights[byte_index];
      const int quantized = int((weight_k & 1) ? (packed >> 4) : (packed & 15)) - 8;
      weight = float(quantized) * scales[weight_row * groups + weight_k / group_size];
    }
    weight_tile[local_position.x][local_position.y] = weight;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint inner = 0; inner < TILE; ++inner) {
      sum += float(activation_tile[local_position.y][inner]) * weight_tile[inner][local_position.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < m && column < n) {
    output[row * n + column] = half(sum);
  }
}


kernel void onyx_q4_gemm_fp16acc32_tile32(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 tile_position [[threadgroup_position_in_grid]],
    uint2 local_position [[thread_position_in_threadgroup]]) {
  threadgroup half activation_tile[32][16];
  threadgroup float weight_tile[16][32];
  const uint local = local_position.y * 16 + local_position.x;
  const uint groups = k / group_size;
  float sums[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

  for (uint tile_k = 0; tile_k < k; tile_k += 16) {
    for (uint index = local; index < 512; index += 256) {
      const uint a_row = index / 16;
      const uint a_k = index % 16;
      const uint row = tile_position.y * 32 + a_row;
      const uint column = tile_k + a_k;
      activation_tile[a_row][a_k] = row < m && column < k ? activation[row * k + column] : half(0.0h);

      const uint weight_row = tile_position.x * 32 + a_row;
      float weight = 0.0f;
      if (weight_row < n && column < k) {
        const uint byte_index = weight_row * (k / 2) + column / 2;
        const uint packed = packed_weights[byte_index];
        const int quantized = int((column & 1) ? (packed >> 4) : (packed & 15)) - 8;
        weight = float(quantized) * scales[weight_row * groups + column / group_size];
      }
      weight_tile[a_k][a_row] = weight;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint inner = 0; inner < 16; ++inner) {
      for (uint row_offset = 0; row_offset < 2; ++row_offset) {
        const float a = float(activation_tile[local_position.y * 2 + row_offset][inner]);
        for (uint column_offset = 0; column_offset < 2; ++column_offset) {
          sums[row_offset][column_offset] += a * weight_tile[inner][local_position.x * 2 + column_offset];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint row_offset = 0; row_offset < 2; ++row_offset) {
    for (uint column_offset = 0; column_offset < 2; ++column_offset) {
      const uint row = tile_position.y * 32 + local_position.y * 2 + row_offset;
      const uint column = tile_position.x * 32 + local_position.x * 2 + column_offset;
      if (row < m && column < n) {
        output[row * n + column] = half(sums[row_offset][column_offset]);
      }
    }
  }
}

// simdgroup-MMA candidate (kernel-agent/q4-mma-v1).
// One simdgroup (32 threads) per threadgroup computes a 32x32 output tile.
// k is stepped by 8: activations are staged as fp16; packed nibbles are
// unpacked, scaled in fp32, rounded once to fp16, and fed to the MMA units.
kernel void onyx_q4_gemm_fp16acc32_mma(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 tile_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint TM = 32;
  constexpr uint TN = 32;
  constexpr uint BK = 8;
  threadgroup half activation_tile[TM][BK];
  threadgroup half weight_tile[BK][TN];
  threadgroup float sum_tile[TM][TN];
  const uint row_base = tile_position.y * TM;
  const uint column_base = tile_position.x * TN;
  const uint groups = k / group_size;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  simdgroup_matrix<half, 8, 8> a_fragments[4];
  simdgroup_matrix<half, 8, 8> b_fragments[4];
  for (uint i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (uint tile_k = 0; tile_k < k; tile_k += BK) {
    // Stage A: thread lane owns row row_base+lane, columns tile_k..+8.
    // Uncoalesced across threads but absorbed by L2; empirically faster
    // than a 32-iteration coalesced staging loop (see kernel-agent notes).
    for (uint j = 0; j < BK; ++j) {
      const uint row = row_base + lane;
      const uint column = tile_k + j;
      activation_tile[lane][j] =
          row < m && column < k ? activation[row * k + column] : half(0.0h);
    }
    // Stage W^T: thread lane owns output column column_base+lane and the 8
    // consecutive packed nibbles tile_k..+8 (one aligned uchar4 when k%8==0).
    const uint column = column_base + lane;
    const bool full_step = (tile_k + BK) <= k;
    const uint base = column < n ? column * (k / 2) + tile_k / 2 : 0u;
    uchar4 bytes;
    if (column < n && full_step) {
      if ((k % 8) == 0) {
        bytes = *reinterpret_cast<device const uchar4*>(packed_weights + base);
      } else {
        bytes = uchar4(
            packed_weights[base + 0], packed_weights[base + 1],
            packed_weights[base + 2], packed_weights[base + 3]);
      }
    } else {
      bytes = uchar4(0, 0, 0, 0);
    }
    for (uint j = 0; j < BK; ++j) {
      const uint column_k = tile_k + j;
      int quantized = 0;
      if (column < n && column_k < k) {
        const uint byte_index = column * (k / 2) + column_k / 2;
        const uint byte = packed_weights[byte_index];
        quantized = int((column_k & 1) ? (byte >> 4) : (byte & 15)) - 8;
      }
      const float scale = column < n && column_k < k
          ? scales[column * groups + column_k / group_size] : 0.0f;
      weight_tile[j][lane] = half(float(quantized) * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 4; ++i) {
      simdgroup_load(a_fragments[i], &activation_tile[i * 8][0], BK);
    }
    for (uint j = 0; j < 4; ++j) {
      simdgroup_load(b_fragments[j], &weight_tile[0][j * 8], TN);
    }
    for (uint i = 0; i < 4; ++i) {
      for (uint j = 0; j < 4; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a_fragments[i], b_fragments[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], &sum_tile[i * 8][j * 8], TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < TM * TN; index += 32) {
    const uint row = row_base + index / TN;
    const uint column = column_base + index % TN;
    if (row < m && column < n) {
      output[row * n + column] = half(sum_tile[index / TN][index % TN]);
    }
  }
}

// kernel-agent/q4-mma-v2 (episode E2): 64x64 output tile, four simdgroups
// per threadgroup, each owning one 32x32 quadrant. Cuts global A/W re-reads
// 4x versus the 32x32 kernel; staging covers the full tile per simdgroup.
kernel void onyx_q4_gemm_fp16acc32_mma64(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    uint2 tile_position [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint TM = 64;
  constexpr uint TN = 64;
  constexpr uint BK = 8;
  const uint sg = tid / 32;
  const uint qrow = sg / 2;
  const uint qcol = sg % 2;
  threadgroup half activation_tile[TM][BK];
  threadgroup half weight_tile[BK][TN];
  threadgroup float sum_tile[TM][TN];
  const uint row_base = tile_position.y * TM;
  const uint column_base = tile_position.x * TN;
  const uint groups = k / group_size;
  simdgroup_matrix<float, 8, 8> acc[4][4];
  simdgroup_matrix<half, 8, 8> a_fragments[4];
  simdgroup_matrix<half, 8, 8> b_fragments[4];
  for (uint i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (uint tile_k = 0; tile_k < k; tile_k += BK) {
    // Stage A: thread tid owns row tid%64 and half of the k-step.
    {
      const uint row = tid % TM;
      const uint ahalf = tid / TM;
      const uint column = tile_k + ahalf * 4;
      for (uint i = 0; i < 4; ++i) {
        activation_tile[row][i + ahalf * 4] =
            row_base + row < m && column + i < k ? activation[(row_base + row) * k + column + i] : half(0.0h);
      }
    }
    // Stage W^T: thread tid owns output column column_base + tid%64 and the
    // four nibbles tile_k + (tid/64)*4 .. +3 (one aligned ushort pair when
    // k%8==0, scalar bytes otherwise).
    {
      const uint column = column_base + tid % TN;
      const uint khalf = tid / TN;
      const bool full_step = (tile_k + BK) <= k;
      const uint base = column < n ? column * (k / 2) + tile_k / 2 + khalf * 2 : 0u;
      ushort pair = 0;
      if (column < n && full_step) {
        if ((k % 8) == 0) {
          pair = *reinterpret_cast<device const ushort*>(packed_weights + base);
        } else {
          pair = uint(packed_weights[base + 0]) | (uint(packed_weights[base + 1]) << 8);
        }
      }
      for (uint i = 0; i < 4; ++i) {
        const uint column_k = tile_k + khalf * 4 + i;
        int quantized = 0;
        float scale = 0.0f;
        if (column < n && column_k < k) {
          uint byte;
          if (full_step) {
            byte = (pair >> ((i / 2) * 8)) & 255u;
          } else {
            byte = packed_weights[column * (k / 2) + column_k / 2];
          }
          quantized = int((column_k & 1) ? (byte >> 4) : (byte & 15)) - 8;
          scale = scales[column * groups + column_k / group_size];
        }
        weight_tile[khalf * 4 + i][tid % TN] = half(float(quantized) * scale);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 4; ++i) {
      simdgroup_load(a_fragments[i], &activation_tile[qrow * 32 + i * 8][0], BK);
    }
    for (uint j = 0; j < 4; ++j) {
      simdgroup_load(b_fragments[j], &weight_tile[0][qcol * 32 + j * 8], TN);
    }
    for (uint i = 0; i < 4; ++i) {
      for (uint j = 0; j < 4; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a_fragments[i], b_fragments[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = 0; i < 4; ++i) {
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], &sum_tile[qrow * 32 + i * 8][qcol * 32 + j * 8], TN);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = tid; index < TM * TN; index += 128) {
    const uint row = row_base + index / TN;
    const uint column = column_base + index % TN;
    if (row < m && column < n) {
      output[row * n + column] = half(sum_tile[index / TN][index % TN]);
    }
  }
}

// kernel-agent/q4-decode-splitk-v1 (episode E1): split-k MMA for small M.
// The mma kernels leave small-m shapes parallelism-starved (672-336
// threadgroups over n only); E8 proved scalar split-k hits the ~8 TFLOPS
// scalar-FMA ceiling, so the k dimension is split across threadgroups and
// each slice runs the simdgroup-MMA staging pattern. Partials land as fp32
// in scratch[split_k][m][n] — every element is written exactly once (no
// atomics, no zero-init, bit-stable) — and a separate reduce encoder sums
// slices and converts to fp16 in the same command buffer.
kernel void onyx_q4_gemm_fp16acc32_splitk(
    device const half* activation [[buffer(0)]],
    device const uchar* packed_weights [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device float* partials [[buffer(8)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group_size [[buffer(7)]],
    constant uint& split_k [[buffer(9)]],
    uint3 tile_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint TM = 16;
  constexpr uint TN = 32;
  constexpr uint BK = 8;
  threadgroup half activation_tile[TM][BK];
  threadgroup half weight_tile[BK][TN];
  threadgroup float sum_tile[8][8];
  const uint row_base = tile_position.y * TM;
  const uint column_base = tile_position.x * TN;
  const uint slice = tile_position.z;
  const uint groups = k / group_size;
  const uint slice_len = (k + split_k - 1) / split_k;
  const uint slice_start = slice * slice_len;
  const uint slice_end = min(slice_start + slice_len, k);
  simdgroup_matrix<float, 8, 8> acc[2][4];
  simdgroup_matrix<half, 8, 8> a_fragments[2];
  simdgroup_matrix<half, 8, 8> b_fragments[4];
  for (uint i = 0; i < 2; ++i) {
    for (uint j = 0; j < 4; ++j) {
      acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }

  for (uint tile_k = slice_start; tile_k < slice_end; tile_k += BK) {
    // Stage A: lane<16 stages row lane, the 8 columns tile_k..+8. m>16 is
    // covered by additional grid-y threadgroups, so only TM rows exist here.
    if (lane < TM) {
      for (uint j = 0; j < BK; ++j) {
        const uint column = tile_k + j;
        activation_tile[lane][j] =
            row_base + lane < m && column < k ? activation[(row_base + lane) * k + column] : half(0.0h);
      }
    }
    // Stage W^T: thread lane owns output column column_base+lane and the 8
    // consecutive nibbles tile_k..+8 (uchar4 when the step is whole and the
    // byte offset stays aligned, scalar loads otherwise).
    const uint column = column_base + lane;
    const bool full_step = (tile_k + BK) <= slice_end;
    const uint base = column < n ? column * (k / 2) + tile_k / 2 : 0u;
    uchar4 bytes = uchar4(0, 0, 0, 0);
    if (column < n && full_step) {
      if ((k % 8) == 0 && (tile_k % 8) == 0) {
        bytes = *reinterpret_cast<device const uchar4*>(packed_weights + base);
      } else {
        bytes = uchar4(
            packed_weights[base + 0], packed_weights[base + 1],
            packed_weights[base + 2], packed_weights[base + 3]);
      }
    }
    for (uint j = 0; j < BK; ++j) {
      const uint column_k = tile_k + j;
      int quantized = 0;
      if (column < n && column_k < slice_end) {
        uint byte;
        if (full_step) {
          byte = bytes[j / 2];
        } else {
          byte = packed_weights[column * (k / 2) + column_k / 2];
        }
        quantized = int((column_k & 1) ? (byte >> 4) : (byte & 15)) - 8;
      }
      const float scale = column < n && column_k < slice_end
          ? scales[column * groups + column_k / group_size] : 0.0f;
      weight_tile[j][lane] = half(float(quantized) * scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 2; ++i) {
      simdgroup_load(a_fragments[i], &activation_tile[i * 8][0], BK);
    }
    for (uint j = 0; j < 4; ++j) {
      simdgroup_load(b_fragments[j], &weight_tile[0][j * 8], TN);
    }
    for (uint i = 0; i < 2; ++i) {
      for (uint j = 0; j < 4; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], a_fragments[i], b_fragments[j], acc[i][j]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Write this slice's fp32 partial tile. Every (slice, row<m, col<n) is
  // covered by exactly one threadgroup, so reduce can sum blindly.
  for (uint i = 0; i < 2; ++i) {
    for (uint j = 0; j < 4; ++j) {
      simdgroup_store(acc[i][j], &sum_tile[0][0], 8);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint index = lane; index < 64; index += 32) {
        const uint row = row_base + i * 8 + index / 8;
        const uint column = column_base + j * 8 + index % 8;
        if (row < m && column < n) {
          partials[size_t(slice) * m * n + size_t(row) * n + column] = sum_tile[index / 8][index % 8];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}

// Sums the split-k partials and converts to fp16. One thread per output.
kernel void onyx_q4_splitk_reduce(
    device const float* partials [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& m [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& split_k [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= m * n) {
    return;
  }
  const uint row = id / n;
  const uint column = id % n;
  float sum = 0.0f;
  for (uint slice = 0; slice < split_k; ++slice) {
    sum += partials[size_t(slice) * m * n + size_t(row) * n + column];
  }
  output[id] = half(sum);
}
