#import "native.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace {

constexpr int64_t kHidden = 5376;
constexpr int64_t kReference16Rows = 33942;
using Clock = std::chrono::steady_clock;
using onyx::Tensor;

Tensor bf16_tensor(id<MTLDevice> device, std::vector<int64_t> shape, uint32_t seed) {
  Tensor tensor{nil, std::move(shape), MPSDataTypeBFloat16};
  tensor.buffer = [device newBufferWithLength:tensor.byte_size() options:MTLResourceStorageModeShared];
  if (!tensor.buffer) throw std::runtime_error("Could not allocate probe tensor");
  auto* values = static_cast<uint16_t*>(tensor.data());
  uint32_t state = seed;
  for (uint64_t i = 0; i < tensor.elements(); ++i) {
    state = state * 1664525U + 1013904223U;
    values[i] = onyx::float_to_bf16((static_cast<int32_t>(state >> 16) - 32768) / 32768.0f);
  }
  return tensor;
}

}  // namespace

int main(int argc, char* argv[]) {
  @autoreleasepool {
    try {
      if (argc < 2 || argc > 3) throw std::runtime_error("Usage: onyx-dit-modulation-probe ROWS [--verify-only]");
      const int64_t rows = std::stoll(argv[1]);
      const bool verify_only = argc == 3 && std::string_view(argv[2]) == "--verify-only";
      if (rows < 1 || (argc == 3 && !verify_only)) throw std::runtime_error("Invalid probe options");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("Metal device unavailable");
      onyx::GraphSession session(device);
      const auto x = bf16_tensor(device, {rows, kHidden}, 1);
      const auto attended = bf16_tensor(device, {rows, kHidden}, 2);
      const auto mlp = bf16_tensor(device, {rows, kHidden}, 3);
      const auto norm1 = bf16_tensor(device, {kHidden}, 4);
      const auto norm2 = bf16_tensor(device, {kHidden}, 5);
      const auto table = bf16_tensor(device, {18, kHidden}, 6);
      Tensor indices{nil, {rows}, MPSDataTypeInt32};
      indices.buffer = [device newBufferWithLength:indices.byte_size() options:MTLResourceStorageModeShared];
      if (!indices.buffer) throw std::runtime_error("Could not allocate modality indices");
      auto* tags = static_cast<int32_t*>(indices.data());
      if (rows == kReference16Rows) {
        constexpr int64_t square_patches = 32 * 32;
        auto* tag = tags;
        for (int image = 0; image < 16; ++image) {
          tag = std::fill_n(tag, image < 9 ? 5 : 6, 1);
          tag = std::fill_n(tag, square_patches + 2, 0);
        }
        tag = std::fill_n(tag, 31, 1);
        tag = std::fill_n(tag, 16 * square_patches, 2);
        std::fill_n(tag, square_patches, 0);
      } else {
        for (int64_t row = 0; row < rows; ++row) tags[row] = (row + 2) % 3;
      }

      const auto apply = [&](bool indexed) -> std::array<Tensor, 4> {
        constexpr double epsilon = 1e-5;
        if (indexed) {
          auto a = session.indexed_modulated_rms_norm(x, norm1, table, indices, 0, epsilon);
          auto r = session.indexed_gated_residual(x, attended, table, indices, 2);
          auto b = session.indexed_modulated_rms_norm(r, norm2, table, indices, 3, epsilon);
          auto y = session.indexed_gated_residual(r, mlp, table, indices, 5);
          return {a, r, b, y};
        }
        const auto parts = session.modulation_parts(table, indices);
        auto a = session.modulated_rms_norm(x, norm1, parts[1], parts[0], epsilon);
        auto r = session.gated_residual(x, attended, parts[2]);
        auto b = session.modulated_rms_norm(r, norm2, parts[4], parts[3], epsilon);
        auto y = session.gated_residual(r, mlp, parts[5]);
        return {a, r, b, y};
      };

      uint64_t unequal_values = 0;
      float max_abs_error = 0;
      @autoreleasepool {
        const auto baseline = apply(false);
        const auto candidate = apply(true);
        for (size_t i = 0; i < baseline.size(); ++i) {
          const auto* left = static_cast<const uint16_t*>(baseline[i].data());
          const auto* right = static_cast<const uint16_t*>(candidate[i].data());
          if (std::memcmp(left, right, baseline[i].byte_size()) == 0) continue;
          for (uint64_t cell = 0; cell < baseline[i].elements(); ++cell) {
            if (left[cell] == right[cell]) continue;
            ++unequal_values;
            max_abs_error = std::max(max_abs_error, std::abs(onyx::bf16_to_float(left[cell]) -
                                                            onyx::bf16_to_float(right[cell])));
          }
        }
      }
      const bool equal = unequal_values == 0;
      std::printf("{\"rows\":%lld,\"bitwise_equal\":%s,\"unequal_values\":%llu,\"max_abs_error\":%g,"
                  "\"baseline_modulation_output_bytes\":%llu,\"candidate_modulation_output_bytes\":0,\"timings_ms\":[",
                  static_cast<long long>(rows), equal ? "true" : "false",
                  static_cast<unsigned long long>(unequal_values), max_abs_error,
                  static_cast<unsigned long long>(6 * x.byte_size()));
      if (equal && !verify_only) {
        // Settle the allocator and GPU clock before starting the alternating samples.
        for (uint32_t warmup = 0; warmup < 12; ++warmup) {
          @autoreleasepool {
            apply(false);
            apply(true);
          }
        }
        constexpr uint32_t repeats = 10;
        for (uint32_t block = 0; block < 3; ++block) {
          for (uint32_t warmup = 0; warmup < 3; ++warmup) {
            @autoreleasepool {
              apply(false);
              apply(true);
            }
          }
          std::printf("%s[", block ? "," : "");
          const std::array<bool, 4> order = {false, true, true, false};
          for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
            for (size_t i = 0; i < order.size(); ++i) {
              // Rotate ABBA to BAAB in the middle block to balance position.
              const bool indexed = order[i] != (block == 1);
              const auto start = Clock::now();
              @autoreleasepool { apply(indexed); }
              const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
              std::printf("%s{\"variant\":\"%s\",\"ms\":%.6f}", (repeat || i) ? "," : "",
                          indexed ? "indexed" : "baseline", ms);
            }
          }
          std::printf("]");
        }
      }
      std::printf("]}\n");
      return equal ? 0 : 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
    }
  }
}
