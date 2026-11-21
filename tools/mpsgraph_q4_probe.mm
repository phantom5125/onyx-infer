#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MPSGraph.h>
#import <MetalPerformanceShadersGraph/MPSGraphMemoryOps.h>
#import <MetalPerformanceShadersGraph/MPSGraphQuantizationOps.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  @autoreleasepool {
    constexpr uint32_t rows = 4;
    constexpr uint32_t columns = 128;
    constexpr uint32_t groups = 2;
    std::vector<uint8_t> packed(rows * columns / 2);
    std::vector<float> scales(rows * groups);
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t group = 0; group < groups; ++group) scales[row * groups + group] = 0.1f * (row + 1) * (group + 1);
      for (uint32_t column = 0; column < columns; ++column) {
        const uint8_t value = (row + column) % 15;
        if (column & 1) packed[row * columns / 2 + column / 2] |= value << 4;
        else packed[row * columns / 2 + column / 2] = value;
      }
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MPSGraph* graph = [MPSGraph new];
    auto input = [graph placeholderWithShape:@[@(rows), @(columns)] dataType:MPSDataTypeUInt4 name:@"q4"];
    auto scale = [graph placeholderWithShape:@[@(rows), @(groups)] dataType:MPSDataTypeFloat32 name:@"scales"];
    auto zero = [graph placeholderWithShape:@[@(rows), @(groups)] dataType:MPSDataTypeUInt4 name:@"zero"];
    auto output = [graph dequantizeTensor:input scaleTensor:scale zeroPointTensor:zero dataType:MPSDataTypeFloat32 name:@"dequantized"];
    auto buffer = [&](const void* data, NSUInteger length) {
      return [device newBufferWithBytes:data length:length options:MTLResourceStorageModeShared];
    };
    auto input_buffer = buffer(packed.data(), packed.size());
    auto scale_buffer = buffer(scales.data(), scales.size() * sizeof(float));
    const std::vector<uint8_t> zero_values(rows * groups / 2, 0x88);
    auto zero_buffer = buffer(zero_values.data(), zero_values.size());
    auto output_buffer = [device newBufferWithLength:rows * columns * sizeof(float) options:MTLResourceStorageModeShared];
    auto input_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer shape:@[@(rows), @(columns)] dataType:MPSDataTypeUInt4];
    auto scale_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:scale_buffer shape:@[@(rows), @(groups)] dataType:MPSDataTypeFloat32];
    auto zero_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:zero_buffer shape:@[@(rows), @(groups)] dataType:MPSDataTypeUInt4];
    auto output_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer shape:@[@(rows), @(columns)] dataType:MPSDataTypeFloat32];
    [graph runWithMTLCommandQueue:queue feeds:@{input: input_data, scale: scale_data, zero: zero_data} targetOperations:nil resultsDictionary:@{output: output_data}];
    const float* actual = static_cast<const float*>(output_buffer.contents);
    float max_error = 0;
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t column = 0; column < columns; ++column) {
        const uint8_t byte = packed[row * columns / 2 + column / 2];
        const int quantized = int((column & 1) ? byte >> 4 : byte & 15) - 8;
        const float expected = quantized * scales[row * groups + column / 64];
        max_error = std::max(max_error, std::abs(actual[row * columns + column] - expected));
      }
    }
    std::printf("{\"operator\":\"mpsgraph.q4.dequantize\",\"max_abs_error\":%.8f,\"correct\":%s}\n", max_error, max_error < 1e-6 ? "true" : "false");
    return max_error < 1e-6 ? 0 : 2;
  }
}
