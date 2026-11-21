// Probe: call mlx::core::quantized_matmul from C++ on onyx q4 data.
// Format bridge: onyx stores unsigned nibble q (0..15) with w = (q-8)*scale.
// MLX quantized format is w = q*scale + bias, so supplying bias = -8*scale
// makes the two mathematically identical without repacking.
#include <mlx/mlx.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

namespace mx = mlx::core;

int main(int argc, char** argv) {
  const uint32_t m = argc > 1 ? std::stoi(argv[1]) : 1024;
  const uint32_t n = argc > 2 ? std::stoi(argv[2]) : 5376;
  const uint32_t k = argc > 3 ? std::stoi(argv[3]) : 5376;
  const uint32_t group = 64;
  @autoreleasepool {
    mx::set_default_device(mx::Device(mx::Device::gpu));
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 0.2f);

    // onyx-format packed weights + scales, with MLX-compatible nibble order
    // (low nibble = lower k index).
    std::vector<uint8_t> packed(size_t(n) * k / 2, 0);
    std::vector<float> scales(size_t(n) * (k / group));
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
    std::vector<uint16_t> activation(size_t(m) * k);
    for (auto& value : activation) {
      const __fp16 half = __fp16(normal(rng));
      value = *reinterpret_cast<const uint16_t*>(&half);
    }

    std::fprintf(stderr, "stage: gen-weights\n");
    // MLX arrays via the copying constructors. Weights are a one-time copy
    // per layer; x costs one m*k*2 bytes copy per call (unified memory).
    std::fprintf(stderr, "stage: w-construct\n");
    // MLX packs 4-bit as uint32 words holding 8 nibbles each: repack our
    // 2-nibbles-per-byte stream, one-time per layer. Little-endian word
    // assembly keeps low nibble = lower k index.
    std::vector<uint32_t> packed32(packed.size() / 4);
    for (size_t index = 0; index < packed32.size(); ++index) {
      packed32[index] = uint32_t(packed[index * 4 + 0]) | (uint32_t(packed[index * 4 + 1]) << 8) |
                        (uint32_t(packed[index * 4 + 2]) << 16) | (uint32_t(packed[index * 4 + 3]) << 24);
    }
    auto w = mx::array(packed32.data(), {int(n), int(k / 8)}, mx::uint32);
    std::fprintf(stderr, "stage: s-construct\n");
    auto s = mx::array(scales.data(), {int(n), int(k / group)}, mx::float32);
    std::vector<float> biases_data(scales.size());
    for (size_t index = 0; index < scales.size(); ++index) biases_data[index] = -8.0f * scales[index];
    auto b = mx::array(biases_data.data(), {int(n), int(k / group)}, mx::float32);
    std::fprintf(stderr, "stage: x-construct\n");
    std::vector<float> activation_f(activation.size());
    for (size_t index = 0; index < activation.size(); ++index) {
      activation_f[index] = float(reinterpret_cast<const __fp16&>(activation[index]));
    }
    auto x = mx::array(activation_f.data(), {int(m), int(k)}, mx::float16);
    std::fprintf(stderr, "stage: arrays-eval\n");
    mx::eval(w, s, b, x);

    std::fprintf(stderr, "stage: mlx-arrays\n");
    auto output = mx::quantized_matmul(x, w, s, b, /*transpose*/ true, /*group_size*/ group, /*bits*/ 4);
    mx::eval(output);
    std::printf("mlx quantized_matmul output (%d,%d)\n", output.shape(0), output.shape(1));

    std::fprintf(stderr, "stage: first-eval\n");
    // Timing.
    for (int warm = 0; warm < 3; ++warm) mx::eval(mx::quantized_matmul(x, w, s, b, true, group, 4));
    std::vector<double> samples;
    for (int iteration = 0; iteration < 10; ++iteration) {
      const auto begin = std::chrono::steady_clock::now();
      mx::eval(mx::quantized_matmul(x, w, s, b, true, group, 4));
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    std::printf("m=%u n=%u k=%u: median %.3f ms (%.1f TFLOPS effective)\n",
        m, n, k, median, 2.0 * m * n * k / (median * 1e-3) / 1e12);

    std::fprintf(stderr, "stage: ref-check\n");
    // CPU reference on sampled outputs (force host sync via fp32 cast).
    auto output_f = mx::astype(output, mx::float32);
    mx::eval(output_f);
    const float* out_bits = output_f.data<float>();
    double max_error = 0.0;
    const size_t checks = 32;
    for (size_t check = 0; check < checks; ++check) {
      const size_t flat = check * (size_t(m) * n / checks);
      const uint32_t row = uint32_t(flat / n);
      const uint32_t column = uint32_t(flat % n);
      float expected = 0.0f;
      for (uint32_t inner = 0; inner < k; ++inner) {
        const float weight = float(quantized[size_t(column) * k + inner]) * scales[size_t(column) * (k / group) + inner / group];
        expected += float(reinterpret_cast<__fp16&>(activation[size_t(row) * k + inner])) * weight;
      }
      const float actual = out_bits[flat];
      max_error = std::max(max_error, std::abs(double(actual) - double(expected)));
    }
    std::printf("max_abs_error %.4g %s\n", max_error, max_error < 0.25 ? "PASS" : "FAIL");
    return max_error < 0.25 ? 0 : 2;
  }
}
