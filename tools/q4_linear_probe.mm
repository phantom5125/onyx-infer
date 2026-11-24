#import "native.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <random>
#include <vector>

// Engine-level integration check for GraphSession::linear with quantized
// weights. CTest runs this check against MPSGraph and the optional FP16
// backends using the same WeightStore format as the engine.
namespace {

void write_store(const std::string& path, const std::vector<uint8_t>& packed,
                 const std::vector<float>& scales, uint32_t n, uint32_t k) {
  const uint64_t packed_bytes = packed.size();
  const uint64_t scale_bytes = scales.size() * 4;
  const uint64_t scale_begin = packed_bytes;
  const uint64_t scale_end = scale_begin + scale_bytes;
  NSDictionary* spec = @{
    @"w.weight": @{@"dtype": @"Q4S64", @"shape": @[@(n), @(k)],
        @"data_offsets": @[@0, @(packed_bytes)],
        @"scale_offsets": @[@(scale_begin), @(scale_end)]},
  };
  NSData* json = [NSJSONSerialization dataWithJSONObject:spec options:0 error:nil];
  const uint64_t length = json.length;
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(&length), 8);
  out.write(static_cast<const char*>(json.bytes), json.length);
  out.write(reinterpret_cast<const char*>(packed.data()), packed.size());
  out.write(reinterpret_cast<const char*>(scales.data()), scales.size() * 4);
}

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "no device\n"); return 1; }
    onyx::GraphSession session(device);

    const uint32_t k = 5376, group = 64;
    // 5376 columns match every Q4 layer of the model. 5360 is not a multiple of 64, which
    // reaches the neural-accelerator kernel that bounds the output columns.
    const uint32_t widths[] = {5376, 5360};
    bool all_pass = true;
    for (const uint32_t n : widths) {
    const size_t packed_count = size_t(n) * k / 2;
    const size_t scale_count = size_t(n) * (k / group);
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 0.2f);
    std::vector<uint8_t> packed(packed_count, 0);
    std::vector<float> scales(scale_count);
    std::vector<int8_t> quantized(size_t(n) * k);
    for (uint32_t row = 0; row < n; ++row) {
      for (uint32_t g = 0; g < k / group; ++g) {
        float maximum = 0.0f;
        for (uint32_t i = 0; i < group; ++i) {
          quantized[size_t(row) * k + g * group + i] = int8_t((rng() % 15) - 7);
          maximum = std::max(maximum, std::abs(float(quantized[size_t(row) * k + g * group + i])));
        }
        const float scale = maximum == 0.0f ? 1e-6f : maximum / 7.0f;
        scales[size_t(row) * (k / group) + g] = scale;
        for (uint32_t i = 0; i < group; ++i) {
          const uint32_t column = g * group + i;
          const uint8_t encoded = uint8_t(quantized[size_t(row) * k + column] + 8);
          const size_t index = size_t(row) * (k / 2) + column / 2;
          if (column & 1) packed[index] |= encoded << 4;
          else packed[index] = encoded;
        }
      }
    }

    const std::string store_path = std::string(NSTemporaryDirectory().UTF8String) + [NSUUID UUID].UUIDString.UTF8String + ".onxpack";
    write_store(store_path, packed, scales, n, k);
    onyx::WeightStore store(store_path, false);

    // FP16 cases cover the optional FP16 backends. m69 covers the multi-row-tile regime of
    // the small-m dispatch bucket. BF16 is the production activation type, with the row
    // counts of a DiT block and of the text encoder.
    struct Case { uint32_t rows; MPSDataType dtype; const char* name; };
    std::vector<Case> cases = {{1024, MPSDataTypeFloat16, "m1024-fp16"}, {69, MPSDataTypeFloat16, "m69-fp16"},
                               {16, MPSDataTypeFloat16, "m16-fp16"},
                               {1093, MPSDataTypeBFloat16, "m1093-bf16"}, {69, MPSDataTypeBFloat16, "m69-bf16"}};
    if (n != 5376) cases = {{1093, MPSDataTypeBFloat16, "n5360-m1093-bf16"}, {16, MPSDataTypeBFloat16, "n5360-m16-bf16"}};
    for (const Case& test : cases) {
      std::vector<uint16_t> activation(size_t(test.rows) * k);
      std::vector<float> activation_float(size_t(test.rows) * k);
      for (size_t index = 0; index < activation.size(); ++index) {
        activation_float[index] = normal(rng);
        if (test.dtype == MPSDataTypeBFloat16) {
          activation[index] = onyx::float_to_bf16(activation_float[index]);
        } else {
          const __fp16 half = __fp16(activation_float[index]);
          activation[index] = *reinterpret_cast<const uint16_t*>(&half);
        }
        // The reference consumes the activations as the kernel receives them.
        activation_float[index] = test.dtype == MPSDataTypeBFloat16 ? onyx::bf16_to_float(activation[index])
                                                                    : float(reinterpret_cast<const __fp16&>(activation[index]));
      }
      auto activation_buffer = [device newBufferWithBytes:activation.data()
                                                   length:activation.size() * 2
                                                  options:MTLResourceStorageModeShared];
      onyx::Tensor x{activation_buffer, {test.rows, k}, test.dtype};
      onyx::Tensor y = session.linear(store, "w", x);
      // Median wall time over 10 calls through the engine entry point.
      {
        std::vector<double> samples;
        for (int timing = 0; timing < 10; ++timing) {
          const auto begin = std::chrono::steady_clock::now();
          onyx::Tensor timed = session.linear(store, "w", x);
          samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        }
        std::sort(samples.begin(), samples.end());
        std::printf("%s: median %.3f ms\n", test.name, samples[samples.size() / 2]);
      }
      if (y.shape != std::vector<int64_t>({test.rows, n}) || y.dtype != test.dtype) {
        std::fprintf(stderr, "%s: bad output shape or dtype\n", test.name);
        return 2;
      }
      const uint16_t* out = static_cast<const uint16_t*>(y.buffer.contents);
      double max_error = 0.0;
      double max_relative = 0.0;
      const size_t checks = 32;
      for (size_t check = 0; check < checks; ++check) {
        const size_t flat = check * (size_t(test.rows) * n / checks);
        const uint32_t row = uint32_t(flat / n);
        const uint32_t column = uint32_t(flat % n);
        // The reference keeps FP32 weights. Every path rounds them to the activation type,
        // so a BF16 path carries a weight rounding error near 2^-9 per product and gets a
        // matching relative tolerance.
        float expected = 0.0f;
        for (uint32_t inner = 0; inner < k; ++inner) {
          const float weight = float(quantized[size_t(column) * k + inner]) * scales[size_t(column) * (k / group) + inner / group];
          expected += activation_float[size_t(row) * k + inner] * weight;
        }
        const float actual = test.dtype == MPSDataTypeBFloat16 ? onyx::bf16_to_float(out[flat])
                                                               : float(reinterpret_cast<const __fp16&>(out[flat]));
        const double absolute = std::abs(double(actual) - double(expected));
        const double relative = std::abs(double(expected)) > 1e-3 ? absolute / std::abs(double(expected)) : 0.0;
        const double tolerance = test.dtype == MPSDataTypeBFloat16 ? 0.05 + 0.0125 * std::abs(double(expected))
                                                                   : 0.05 + 0.005 * std::abs(double(expected));
        if (!std::isfinite(actual) || absolute > tolerance) all_pass = false;
        max_error = std::max(max_error, absolute);
        max_relative = std::max(max_relative, relative);
      }
      std::printf("%s: max_abs_error %.4g max_relative_error %.4g %s\n", test.name, max_error, max_relative, all_pass ? "PASS" : "FAIL");
    }
    std::remove(store_path.c_str());
    }
    return all_pass ? 0 : 2;
  }
}
