#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>

#import "native.hpp"
#import <MetalPerformanceShadersGraph/MPSGraphMatrixMultiplicationOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphTensorShapeOps.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace {

uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float from_bf16(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool native_view_test(id<MTLDevice> device) {
  float values[16];
  for (uint32_t index = 0; index < 16; ++index) values[index] = static_cast<float>(index + 1);
  auto buffer = [device newBufferWithBytes:values length:sizeof(values) options:MTLResourceStorageModeShared];
  onyx::GraphSession session(device);
  onyx::Tensor input{buffer, {4, 4}, MPSDataTypeFloat32};
  auto view = session.reshape(session.slice(input, 0, 1, 2), {1, 2, 4});
  auto output = session.add(view, onyx::Tensor{buffer, {1}, MPSDataTypeFloat32});
  if (view.byte_offset != 16 || output.byte_size() != 32) return false;
  const float* actual = static_cast<const float*>(output.buffer.contents);
  for (uint32_t index = 0; index < 8; ++index) {
    if (actual[index] != static_cast<float>(index + 6)) return false;
  }
  return true;
}

}  // namespace

int main() {
  @autoreleasepool {
    const std::vector<float> x = {1, 2, 3, 4, 5, 6};
    const std::vector<float> w = {0.5, -1, 2, 1, 0, -0.5, 1, 1.5, -2, 0.25, 2, 1};
    std::vector<uint16_t> xb, wb;
    for (float value : x) xb.push_back(bf16(value));
    for (float value : w) wb.push_back(bf16(value));
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraphDevice* graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
    MPSGraph* graph = [MPSGraph new];
    MPSGraphTensor* input = [graph placeholderWithShape:@[@2, @3] dataType:MPSDataTypeBFloat16 name:@"x"];
    MPSGraphTensor* weights = [graph placeholderWithShape:@[@4, @3] dataType:MPSDataTypeBFloat16 name:@"w"];
    MPSGraphTensor* transposed = [graph transposeTensor:weights permutation:@[@1, @0] name:nil];
    MPSGraphTensor* output = [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:transposed name:@"y"];
    auto input_data = [[MPSGraphTensorData alloc] initWithDevice:graph_device data:[NSData dataWithBytes:xb.data() length:xb.size() * 2] shape:@[@2, @3] dataType:MPSDataTypeBFloat16];
    auto weight_data = [[MPSGraphTensorData alloc] initWithDevice:graph_device data:[NSData dataWithBytes:wb.data() length:wb.size() * 2] shape:@[@4, @3] dataType:MPSDataTypeBFloat16];
    id<MTLBuffer> output_buffer = [device newBufferWithLength:8 * 2 options:MTLResourceStorageModeShared];
    auto output_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer shape:@[@2, @4] dataType:MPSDataTypeBFloat16];
    [graph runWithMTLCommandQueue:queue feeds:@{input: input_data, weights: weight_data} targetOperations:nil resultsDictionary:@{output: output_data}];
    const uint16_t* values = static_cast<const uint16_t*>(output_buffer.contents);
    const float expected[] = {4.5, -0.5, -2, 7.25, 9, 1, -0.5, 17};
    float max_error = 0;
    for (int index = 0; index < 8; ++index) {
      max_error = std::max(max_error, std::abs(from_bf16(values[index]) - expected[index]));
    }
    std::vector<float> q(16), k(16), v(16);
    for (int index = 0; index < 16; ++index) {
      q[index] = (index % 7 - 3) * 0.2f;
      k[index] = (index % 5 - 2) * 0.25f;
      v[index] = (index % 3 - 1) * 0.5f;
    }
    auto make_tensor = [&](const std::vector<float>& values) {
      std::vector<uint16_t> encoded;
      for (float value : values) encoded.push_back(bf16(value));
      auto buffer = [device newBufferWithBytes:encoded.data() length:encoded.size() * sizeof(uint16_t)
                                     options:MTLResourceStorageModeShared];
      return onyx::Tensor{buffer, {1, 2, 2, 4}, MPSDataTypeBFloat16};
    };
    onyx::GraphSession session(device);
    const onyx::Tensor query = make_tensor(q), key = make_tensor(k), value = make_tensor(v);
    float max_attention_error = 0.0f;
    for (bool causal : {false, true}) {
      const onyx::Tensor attention = session.attention(query, key, value, causal, 0.5);
      const uint16_t* actual_attention = static_cast<const uint16_t*>(attention.data());
      for (int head = 0; head < 2; ++head) {
        for (int row = 0; row < 2; ++row) {
          float scores[2];
          for (int column = 0; column < 2; ++column) {
            float score = 0;
            for (int dim = 0; dim < 4; ++dim) {
              const int qi = head * 8 + row * 4 + dim;
              const int ki = head * 8 + column * 4 + dim;
              score += from_bf16(bf16(q[qi])) * from_bf16(bf16(k[ki]));
            }
            scores[column] = score * 0.5f;
          }
          const float maximum = causal && row == 0 ? scores[0] : std::max(scores[0], scores[1]);
          const float e0 = std::exp(scores[0] - maximum);
          const float e1 = causal && row == 0 ? 0.0f : std::exp(scores[1] - maximum);
          for (int dim = 0; dim < 4; ++dim) {
            float expected_attention = 0;
            for (int column = 0; column < 2; ++column) {
              const float probability = (column == 0 ? e0 : e1) / (e0 + e1);
              expected_attention += probability * from_bf16(bf16(v[head * 8 + column * 4 + dim]));
            }
            const float actual = from_bf16(actual_attention[head * 8 + row * 4 + dim]);
            if (!std::isfinite(actual)) return 2;
            max_attention_error = std::max(max_attention_error, std::abs(actual - expected_attention));
          }
        }
      }
    }
    const bool view_correct = native_view_test(device);
    const bool correct = max_error < 0.05 && max_attention_error < 0.05 && view_correct;
    std::printf("{\"matmul_max_abs_error\":%.8f,\"attention_max_abs_error\":%.8f,\"view_correct\":%s,\"correct\":%s}\n", max_error, max_attention_error, view_correct ? "true" : "false", correct ? "true" : "false");
    return correct ? 0 : 2;
  }
}
