// Compare the GraphSession Steel affine QMM path with a CPU reference.
// Input patterns are LCG (default), a unit vector, and all ones.
// Usage: <metallib> <onxpack> [m] [steel|mpsgraph] [unit|ones|lcg] [dump]
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "native.hpp"

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { std::fprintf(stderr, "usage: <metallib> <onxpack> [m] [steel|mpsgraph] [unit|ones|lcg] [dump]\n"); return 2; }
    FILE* pack_file = std::fopen(argv[2], "rb");
    if (!pack_file) { std::printf("{\"test\":\"linear_steel_probe\",\"status\":\"skip\"}\n"); return 0; }
    std::fclose(pack_file);
    const std::string kernel_mode = argc < 5 ? "steel" : std::string(argv[4]);
    setenv("ONYX_Q4_LINEAR", kernel_mode.c_str(), 1);
    setenv("ONYX_METALLIB", argv[1], 1);
    const uint32_t m = argc > 3 ? uint32_t(std::atoi(argv[3])) : 69;
    const std::string x_mode = argc >= 6 ? std::string(argv[5]) : "lcg";

    try {
      const std::string pack = argv[2];
      // Optional tensor prefix. The default targets the text encoder Q projection.
      const std::string prefix = argc >= 7 ? std::string(argv[6])
                                           : "model.language_model.layers.0.self_attn.q_proj";
      onyx::WeightStore store(pack);
      const uint32_t k = uint32_t(store.shape(prefix + ".weight")[1]);
      const uint32_t n = uint32_t(store.shape(prefix + ".weight")[0]);
      (void)n;
      (void)n;

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      onyx::GraphSession session(device);

      std::vector<float> x_host(size_t(m) * k, 0.0f);
      if (x_mode == "unit") x_host[0] = 1.0f;
      else if (x_mode == "ones") { for (auto& v : x_host) v = 1.0f; }
      else { uint32_t st = 12345; for (auto& v : x_host) { st = st * 1664525u + 1013904223u; v = float(int(st >> 16) % 2000 - 1000) / 1000.0f; } }
      std::vector<uint16_t> x_bf16(x_host.size());
      for (size_t i = 0; i < x_host.size(); ++i) x_bf16[i] = onyx::float_to_bf16(x_host[i]);
      id<MTLBuffer> x_buffer = [device newBufferWithBytes:x_bf16.data() length:x_bf16.size() * 2 options:MTLResourceStorageModeShared];
      onyx::Tensor x{x_buffer, std::vector<int64_t>{m, k}, MPSDataTypeBFloat16};

      const onyx::Tensor y = session.linear(store, prefix, x);
      // The Steel wrapper converts FP16 kernel output back to BF16.
      const uint16_t* y_bits = static_cast<const uint16_t*>(y.buffer.contents);
      std::printf("y[:8] = ");
      for (int c = 0; c < 8; ++c) std::printf("%.4f ", double(onyx::bf16_to_float(y_bits[c])));
      std::printf("  (mode=%s m=%u)\n", x_mode.c_str(), m);

      // Accumulate CPU reference products in float using (nibble - 8) * scale.
      const onyx::Tensor w_packed = store.tensor(prefix + ".weight");
      const onyx::Tensor w_scales = store.scale(prefix + ".weight");
      const float* scales = static_cast<const float*>(w_scales.buffer.contents);
      int bad = 0;
      double max_err = 0.0;
      const uint32_t group = 64;
      for (uint32_t r = 0; r < m && bad < 4; ++r)
        for (uint32_t c = 0; c < n && bad < 4; ++c) {
          double expv = 0.0;
          double abs_sum = 0.0;  // Sum absolute products to bound FP16 rounding under cancellation.
          for (uint32_t x_idx = 0; x_idx < k; ++x_idx) {
            const uint8_t byte = static_cast<const uint8_t*>(w_packed.buffer.contents)[size_t(c) * (k / 2) + x_idx / 2];
            const int q = int((x_idx & 1) ? (byte >> 4) : (byte & 15)) - 8;
            const double term = double(x_host[size_t(r) * k + x_idx]) * double(q * scales[size_t(c) * (k / group) + x_idx / group]);
            expv += term;
            abs_sum += std::abs(term);
          }
          const double got = double(onyx::bf16_to_float(y_bits[size_t(r) * n + c]));
          const double err = std::abs(got - expv);
          max_err = std::max(max_err, err);
          // Tolerance combines an absolute floor, 2% relative error and an FP16 cancellation term.
          if (err > 0.05 + 0.02 * std::abs(expv) + 5e-4 * abs_sum) {
            if (bad < 4) std::printf("bad [%u][%u] got=%.4f exp=%.4f\n", r, c, got, expv);
            ++bad;
          }
        }
      std::printf("{\"test\":\"linear_steel_probe\",\"status\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u,\"max_abs_error\":%.5f}\n",
                  bad ? "fail" : "pass", m, n, k, max_err);
      return bad ? 1 : 0;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "linear-steel-probe: %s\n", error.what());
      return 1;
    }
  }
}
