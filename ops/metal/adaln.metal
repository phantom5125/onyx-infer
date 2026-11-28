#include <metal_stdlib>

using namespace metal;

struct GateParameters {
  uint vectors_per_row;
  uint gate_part;
};

kernel void onyx_adaln_gate(
    device bfloat4* residual [[buffer(0)]],
    const device bfloat4* update [[buffer(1)]],
    const device bfloat4* table [[buffer(2)]],
    const device int* modality [[buffer(3)]],
    constant GateParameters& params [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
  const uint cell = position.y * params.vectors_per_row + position.x;
  const uint gate_row = uint(modality[position.y]) * 6 + params.gate_part;
  const uint gate = gate_row * params.vectors_per_row + position.x;
  // MPSGraph rounds the product to BF16 before adding it to the residual.
  const bfloat4 scaled = bfloat4(float4(update[cell]) * float4(table[gate]));
  residual[cell] = bfloat4(float4(residual[cell]) + float4(scaled));
}
