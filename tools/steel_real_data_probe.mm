// Bind the Steel kernel directly to real Onyx-packed weights.
// Compare the FP16 kernel with a CPU dequantization and accumulation reference.
// This isolates kernel behavior from GraphSession wrapping.
// Usage: steel-real-data-probe <metallib> <onxpack> [m]. The default is 32.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { std::fprintf(stderr, "usage: <metallib> <onxpack> [m]\n"); return 2; }
    FILE* pack_file = std::fopen(argv[2], "rb");
    if (!pack_file) { std::printf("{\"status\":\"skip\"}\n"); return 0; }
    const uint32_t m = argc > 3 ? uint32_t(std::atoi(argv[3])) : 32;

    std::fseek(pack_file, 0, SEEK_END);
    const long file_size = std::ftell(pack_file);
    std::fseek(pack_file, 0, SEEK_SET);
    std::vector<char> raw(file_size);
    if (std::fread(raw.data(), 1, file_size, pack_file) != size_t(file_size)) { std::fclose(pack_file); return 1; }
    std::fclose(pack_file);
    if (std::memcmp(raw.data(), "ONYXPK01", 8) != 0) { std::printf("{\"status\":\"fail\",\"reason\":\"bad pack\"}\n"); return 1; }
    const long long header_len = *reinterpret_cast<long long*>(raw.data() + 8);
    const long long data_begin = 16 + header_len;
    const std::string header_json(raw.data() + 16, header_len);

    auto extract_offsets = [&](const std::string& key) {
      std::pair<long long, long long> result{0, 0};
      size_t pos = header_json.find("\"" + key + "\"");
      if (pos == std::string::npos) return result;
      pos = header_json.find('[', pos);
      const size_t end = header_json.find(']', pos);
      size_t start = pos + 1;
      for (int i = 0; i < 2; ++i) {
        size_t comma = header_json.find(',', start);
        if (comma == std::string::npos || comma > end) comma = end;
        (&result.first)[i] = std::atoll(header_json.substr(start, comma - start).c_str());
        start = comma + 1;
      }
      return result;
    };
    // Read dimensions from the tensor header when a report is unavailable.
    size_t shape_pos = header_json.find("\"shape\"");
    size_t bracket = header_json.find('[', shape_pos);
    const size_t shape_end = header_json.find(']', bracket);
    const long long n_ll = std::atoll(header_json.substr(bracket + 1, header_json.find(',', bracket) - bracket - 1).c_str());
    const long long k_ll = std::atoll(header_json.substr(header_json.find(',', bracket) + 1, shape_end - header_json.find(',', bracket) - 1).c_str());
    (void)shape_pos; (void)bracket; (void)shape_end;
    std::fprintf(stderr, "parse: n=%lld k=%lld\n", n_ll, k_ll);
    const auto data_offsets = extract_offsets("data_offsets");
    std::fprintf(stderr, "parse: data_offsets=(%lld,%lld)\n", data_offsets.first, data_offsets.second);
    const auto scale_offsets = extract_offsets("scale_offsets");
    std::fprintf(stderr, "parse: scale_offsets=(%lld,%lld)\n", scale_offsets.first, scale_offsets.second);
    const uint32_t n = uint32_t(n_ll);
    const uint32_t k = uint32_t(k_ll);
    const uint32_t group = 64;
    const uint32_t groups_per_row = k / group;
    std::fprintf(stderr, "debug: n=%u k=%u data=(%lld,%lld) scale=(%lld,%lld) header_len=%lld\n", n, k, data_offsets.first, data_offsets.second, scale_offsets.first, scale_offsets.second, header_len);
    if (!n || !k || k % 32 || n % 32) { std::printf("{\"status\":\"fail\",\"reason\":\"shape\"}\n"); return 1; }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if (!library) { std::printf("{\"status\":\"fail\",\"reason\":\"metallib\"}\n"); return 1; }
    id<MTLFunction> function = [library newFunctionWithName:@"affine_qmm_t_float16_t_gs_64_b_4_hgs_0_alN_0_batch_0_bm_32_bn_32"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Construct K-axis diagnostic activations and load the packed scales.
    std::vector<__fp16> act(size_t(m) * k, __fp16(0));
    if (argc > 4 && std::string(argv[4]) == "unit") act[0] = __fp16(1.0f);  // unit-x
    else for (uint32_t r = 0; r < m; ++r)
      for (uint32_t c = 0; c < k; ++c)
        act[size_t(r) * k + c] = __fp16((c % 2) ? -1.0f : 1.0f);
    const char* packed_bytes = raw.data() + data_begin + data_offsets.first;
    const float* scale_floats = reinterpret_cast<const float*>(raw.data() + data_begin + scale_offsets.first);
    std::vector<__fp16> scales_half(size_t(n) * groups_per_row);
    std::vector<__fp16> biases_half(size_t(n) * groups_per_row);
    for (size_t i = 0; i < scales_half.size(); ++i) {
      scales_half[i] = __fp16(scale_floats[i]);
      biases_half[i] = __fp16(-8.0f * scale_floats[i]);
    }

    id<MTLBuffer> act_buffer = [device newBufferWithBytes:act.data() length:act.size() * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> w_buffer = [device newBufferWithBytes:packed_bytes length:size_t(n) * (k / 2) options:MTLResourceStorageModeShared];
    id<MTLBuffer> scale_buffer = [device newBufferWithBytes:scales_half.data() length:scales_half.size() * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bias_buffer = [device newBufferWithBytes:biases_half.data() length:biases_half.size() * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buffer = [device newBufferWithLength:size_t(m) * n * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> meta = [device newBufferWithLength:128 options:MTLResourceStorageModeShared];  // Placeholder batch metadata for the unbatched kernel.

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:pipeline];
    {
      const uint8_t* wb = static_cast<const uint8_t*>(w_buffer.contents);
      std::fprintf(stderr, "direct debug: w bytes[:8] =");
      for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %02x", wb[i]);
      const uint16_t* sb = static_cast<const uint16_t*>(scale_buffer.contents);
      std::fprintf(stderr, "  s_fp16[:4] =");
      for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %04x", sb[i]);
      const uint16_t* bb = static_cast<const uint16_t*>(bias_buffer.contents);
      std::fprintf(stderr, "  b_fp16[:4] =");
      for (int i = 0; i < 4; ++i) std::fprintf(stderr, " %04x", bb[i]);
      std::fprintf(stderr, "\n");
    }
    [en setBuffer:w_buffer offset:0 atIndex:0];
    [en setBuffer:scale_buffer offset:0 atIndex:1];
    [en setBuffer:bias_buffer offset:0 atIndex:2];
    [en setBuffer:act_buffer offset:0 atIndex:3];
    [en setBuffer:out_buffer offset:0 atIndex:4];
    int k_i = int(k), n_i = int(n), m_i = int(m);
    [en setBytes:&k_i length:4 atIndex:5];
    [en setBytes:&n_i length:4 atIndex:6];
    [en setBytes:&m_i length:4 atIndex:7];
    [en setBuffer:meta offset:0 atIndex:8];
    [en setBuffer:meta offset:4 atIndex:9];
    [en setBuffer:meta offset:20 atIndex:10];
    [en setBuffer:meta offset:0 atIndex:11];
    [en setBuffer:meta offset:4 atIndex:12];
    [en setBuffer:meta offset:20 atIndex:13];
    [en setBuffer:meta offset:20 atIndex:14];
    [en setBuffer:meta offset:20 atIndex:15];
    [en dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (m + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) { std::printf("{\"status\":\"fail\",\"reason\":\"command\"}\n"); return 1; }

    // Compute per-column CPU reference values for the input pattern.
    const __fp16* out = reinterpret_cast<const __fp16*>(out_buffer.contents);
    int bad = 0;
    double max_err = 0.0;
    for (uint32_t r = 0; r < m; ++r)
      for (uint32_t c = 0; c < n; ++c) {
        double expv = 0.0;
        for (uint32_t x_idx = 0; x_idx < k; ++x_idx) {
          const uint8_t byte = uint8_t(packed_bytes[size_t(c) * (k / 2) + x_idx / 2]);
          const int q = int((x_idx & 1) ? (byte >> 4) : (byte & 15)) - 8;
          expv += double(act[size_t(r) * k + x_idx]) * double(q * scale_floats[size_t(c) * groups_per_row + x_idx / group]);
        }
        const double got = double(out[size_t(r) * n + c]);
        const double err = std::abs(got - expv);
        max_err = std::max(max_err, err);
        if (err > 0.05 + 0.005 * std::abs(expv)) {
          if (bad < 6) std::printf("bad [%u][%u] got=%.4f exp=%.4f\n", r, c, got, expv);
          ++bad;
        }
      }
    std::printf("unit-x row0[:8]: ");
    for (int c = 0; c < 8; ++c) std::printf("%.4f ", double(out[c]));
    std::printf("\n(Expected: the first eight values of NumPy W[:,0].)\n");
    std::printf("unit-x row0[:8]: ");
    for (int c = 0; c < 8; ++c) std::printf("%.4f ", double(out[c]));
    std::printf("\n(Expected NumPy W[:,0] = [-0.0272, 0.0391, -0.0083, ...])\n");
    std::printf("{\"status\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u,\"max_abs_error\":%.5f}\n",
                bad ? "fail" : "pass", m, n, k, max_err);
    return bad ? 1 : 0;
  }
}
