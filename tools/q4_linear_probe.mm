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

    const uint32_t n = 5376, k = 5376, group = 64;
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

    struct Case { uint32_t rows; const char* name; };
    const Case cases[] = {{1024, "m1024"}, {16, "m16"}};
    bool all_pass = true;
    for (const Case& test : cases) {
      std::vector<uint16_t> activation(size_t(test.rows) * k);
      for (auto& value : activation) {
        const __fp16 half = __fp16(normal(rng));
        value = *reinterpret_cast<const uint16_t*>(&half);
      }
      auto activation_buffer = [device newBufferWithBytes:activation.data()
                                                   length:activation.size() * 2
                                                  options:MTLResourceStorageModeShared];
      onyx::Tensor x{activation_buffer, {test.rows, k}, MPSDataTypeFloat16};
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
      if (y.shape != std::vector<int64_t>({test.rows, n})) {
        std::fprintf(stderr, "%s: bad output shape\n", test.name);
        return 2;
      }
      const __fp16* out = static_cast<const __fp16*>(y.buffer.contents);
      double max_error = 0.0;
      const size_t checks = 32;
      for (size_t check = 0; check < checks; ++check) {
        const size_t flat = check * (size_t(test.rows) * n / checks);
        const uint32_t row = uint32_t(flat / n);
        const uint32_t column = uint32_t(flat % n);
        float expected = 0.0f;
        for (uint32_t inner = 0; inner < k; ++inner) {
          const float weight = float(quantized[size_t(column) * k + inner]) * scales[size_t(column) * (k / group) + inner / group];
          expected += float(reinterpret_cast<__fp16&>(activation[size_t(row) * k + inner])) * weight;
        }
        const float actual = out[flat];
        const double absolute = std::abs(double(actual) - double(expected));
        if (!std::isfinite(actual) || absolute > 0.05 + 0.005 * std::abs(double(expected))) all_pass = false;
        max_error = std::max(max_error, absolute);
      }
      std::printf("%s: max_abs_error %.4g %s\n", test.name, max_error, all_pass ? "PASS" : "FAIL");
    }
    std::remove(store_path.c_str());
    return all_pass ? 0 : 2;
  }
}
