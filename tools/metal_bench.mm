#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

struct VariantConfig {
  std::string function_name;
  MTLSize threads;          // threads per threadgroup
  uint32_t tile_m;          // output rows per threadgroup
  uint32_t tile_n;          // output columns per threadgroup
};

// Each variant defines its threadgroup and tile dimensions.
// tile16 and tile32 use 16×16 threads. MMA variants use simdgroups.
VariantConfig variant_config(const std::string& variant) {
  if (variant == "tile16") return {"onyx_q4_gemm_fp16acc32", MTLSizeMake(16, 16, 1), 16, 16};
  if (variant == "tile32") return {"onyx_q4_gemm_fp16acc32_tile32", MTLSizeMake(16, 16, 1), 32, 32};
  if (variant == "simd") return {"onyx_q4_gemm_simd_bn32_bk32", MTLSizeMake(32, 2, 2), 32, 32};
  if (variant == "mlx-steel") return {"affine_qmm_t_float16_t_gs_64_b_4_hgs_0_alN_0_batch_0_bm_32_bn_32", MTLSizeMake(32, 2, 2), 32, 32};  // Four simdgroups cover four quadrants.
  // See benchmarks/kernel-experiments/q4-mma-v1.md for the MMA comparisons.
  if (variant == "mma") return {"onyx_q4_gemm_fp16acc32_mma", MTLSizeMake(32, 1, 1), 32, 32};    // 1 simdgroup/TG，32x32 tile
  if (variant == "mma64") return {"onyx_q4_gemm_fp16acc32_mma64", MTLSizeMake(128, 1, 1), 64, 64};  // 4 simdgroup，64x64 tile
  if (variant.rfind("splitk", 0) == 0) return {"onyx_q4_gemm_fp16acc32_splitk", MTLSizeMake(32, 1, 1), 16, 32};  // split-k MMA for small m (TM=16, TN=32); k slices parsed from the name.
  throw std::runtime_error("Unknown kernel variant: " + variant);
}

// Split-k slice count for splitkN variants (e.g. "splitk4" -> 4); 0 otherwise.
uint32_t split_k_for_variant(const std::string& variant) {
  if (variant.rfind("splitk", 0) == 0) return uint32_t(std::stoul(variant.substr(6)));
  return 0;
}

}  // namespace

namespace {

struct Options {
  uint32_t m = 16;
  uint32_t n = 128;
  uint32_t k = 256;
  uint32_t group = 64;
  uint32_t iterations = 5;
  std::string variant = "tile16";
  std::string fixture;
  std::string output;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    std::string name = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("Missing value for " + name);
    }
    std::string value = argv[++index];
    if (name == "--m") options.m = std::stoul(value);
    else if (name == "--n") options.n = std::stoul(value);
    else if (name == "--k") options.k = std::stoul(value);
    else if (name == "--group") options.group = std::stoul(value);
    else if (name == "--iterations") options.iterations = std::stoul(value);
    else if (name == "--variant") options.variant = value;
    else if (name == "--fixture") options.fixture = value;
    else if (name == "--output") options.output = value;
    else throw std::runtime_error("Unknown option: " + name);
  }
  if (!options.m || !options.n || !options.k || !options.group || options.k % 2 || options.k % options.group || !options.iterations) {
    throw std::runtime_error("Require positive dimensions, iterations, and K divisible by group size");
  }
  variant_config(options.variant);
  if ((options.variant == "simd" && (options.group % 32 != 0 || options.n % 32 != 0)) ||
      (options.variant == "mlx-steel" && options.group != 64)) {
    throw std::runtime_error("Unsupported group size or tile shape for " + options.variant);
  }
  return options;
}

template <typename T>
std::vector<T> read_file(const std::string& path, size_t count) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || static_cast<size_t>(stream.tellg()) != count * sizeof(T)) {
    throw std::runtime_error("Invalid fixture file: " + path);
  }
  stream.seekg(0);
  std::vector<T> values(count);
  stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
  if (!stream) throw std::runtime_error("Could not read fixture: " + path);
  return values;
}

void write_json(const std::string& path, const std::string& json) {
  if (path.empty()) return;
  std::ofstream stream(path);
  stream << json << '\n';
  if (!stream) throw std::runtime_error("Could not write JSON: " + path);
}

float half_to_float(__fp16 value) {
  return static_cast<float>(value);
}

std::string json_escape(const std::string& value) {
  std::string result;
  for (char character : value) {
    if (character == '\\' || character == '"') result += '\\';
    if (character == '\n') result += "\\n";
    else result += character;
  }
  return result;
}

std::string json_numbers(const std::vector<double>& values, double scale) {
  std::ostringstream output;
  output << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index] * scale;
  }
  output << ']';
  return output.str();
}

std::string json_arguments(int argc, char** argv) {
  std::ostringstream output;
  output << '[';
  for (int index = 0; index < argc; ++index) {
    if (index) output << ',';
    output << '"' << json_escape(argv[index]) << '"';
  }
  output << ']';
  return output.str();
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      const Options options = parse_options(argc, argv);
      const size_t activation_count = size_t(options.m) * options.k;
      const size_t weight_count = size_t(options.n) * options.k;
      const size_t packed_count = weight_count / 2;
      const size_t scale_count = size_t(options.n) * (options.k / options.group);
      const size_t output_count = size_t(options.m) * options.n;
      std::mt19937 rng(42);
      std::normal_distribution<float> normal(0.0f, 0.2f);
      std::vector<__fp16> activation(activation_count);
      for (auto& value : activation) value = __fp16(normal(rng));
      std::vector<uint8_t> packed;
      std::vector<float> scales;
      if (!options.fixture.empty()) {
        activation = read_file<__fp16>(options.fixture + "/input.f16", activation_count);
        packed = read_file<uint8_t>(options.fixture + "/weights.q4", packed_count);
        scales = read_file<float>(options.fixture + "/scales.f32", scale_count);
      } else {
        std::vector<float> weights(weight_count);
        for (auto& value : weights) value = normal(rng);
        packed.resize(packed_count);
        scales.resize(scale_count);
        for (uint32_t row = 0; row < options.n; ++row) {
          for (uint32_t group = 0; group < options.k / options.group; ++group) {
            float maximum = 0.0f;
            for (uint32_t inner = 0; inner < options.group; ++inner) {
              maximum = std::max(maximum, std::abs(weights[row * options.k + group * options.group + inner]));
            }
            const float scale = maximum / 7.0f;
            scales[row * (options.k / options.group) + group] = scale;
            for (uint32_t inner = 0; inner < options.group; ++inner) {
              const uint32_t column = group * options.group + inner;
              const int quantized = scale == 0 ? 0 : std::clamp(int(std::nearbyint(weights[row * options.k + column] / scale)), -7, 7);
              const uint8_t encoded = uint8_t(quantized + 8);
              const size_t index = row * (options.k / 2) + column / 2;
              if (column & 1) packed[index] |= encoded << 4;
              else packed[index] = encoded;
            }
          }
        }
      }

      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (!device) throw std::runtime_error("No Metal device");
      NSError* error = nil;
      NSURL* library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:ONYX_METALLIB_PATH]];
      id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
      if (!library) throw std::runtime_error("Could not load metallib: " + std::string(error.localizedDescription.UTF8String));
      const VariantConfig config = variant_config(options.variant);
      id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:config.function_name.c_str()]];
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
      if (!pipeline) throw std::runtime_error("Could not create pipeline: " + std::string(error.localizedDescription.UTF8String));
      id<MTLCommandQueue> queue = [device newCommandQueue];
      // Pad M to 32 rows for the SIMD and Steel dispatch fixtures.
      const bool padded_variant = options.variant.rfind("simd", 0) == 0 || options.variant == "mlx-steel";
      const uint32_t padded_m = padded_variant ? (options.m + 31) / 32 * 32 : options.m;
      std::vector<__fp16> padded_activation(size_t(padded_m) * options.k, __fp16(0));
      for (uint32_t row = 0; row < options.m; ++row) {
        std::copy_n(activation.data() + size_t(row) * options.k, options.k, padded_activation.data() + size_t(row) * options.k);
      }
      auto activation_buffer = [device newBufferWithBytes:padded_activation.data() length:padded_activation.size() * sizeof(__fp16) options:MTLResourceStorageModeShared];
      auto packed_buffer = [device newBufferWithBytes:packed.data() length:packed.size() options:MTLResourceStorageModeShared];
      auto scale_buffer = [device newBufferWithBytes:scales.data() length:scales.size() * sizeof(float) options:MTLResourceStorageModeShared];
      auto output_buffer = [device newBufferWithLength:size_t(padded_m) * options.n * sizeof(__fp16) options:MTLResourceStorageModeShared];
      if (!activation_buffer || !packed_buffer || !scale_buffer || !output_buffer) throw std::runtime_error("Metal allocation failed");

      // Split-k variants accumulate fp32 partials in scratch (one grid-z
      // layer per slice); the reduce encoder sums them into the fp16 output.
      const uint32_t split_k = split_k_for_variant(options.variant);
      id<MTLBuffer> partials_buffer = nil;
      id<MTLComputePipelineState> reduce_pipeline = nil;
      if (split_k > 0) {
        partials_buffer = [device newBufferWithLength:size_t(split_k) * options.m * options.n * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLFunction> reduce_function = [library newFunctionWithName:@"onyx_q4_splitk_reduce"];
        reduce_pipeline = [device newComputePipelineStateWithFunction:reduce_function error:&error];
        if (!partials_buffer || !reduce_pipeline) throw std::runtime_error("Metal allocation failed (splitk partials/reduce)");
      }

      // Steel uses FP16 scales and affine biases.
      // The bias term maps Onyx (nibble - 8) * scale to the MLX affine form.
      const bool steel_variant = options.variant == "mlx-steel";
      std::vector<__fp16> steel_scales, steel_biases;
      id<MTLBuffer> steel_scale_buffer = nil;
      id<MTLBuffer> steel_bias_buffer = nil;
      id<MTLBuffer> priv_act = nil, priv_w = nil, priv_sc = nil, priv_bi = nil, priv_out = nil;
      int32_t batch_ndims_zero = 0;
      int32_t batch_shape_dummy[4] = {0, 0, 0, 0};
      int64_t batch_strides_dummy[4] = {0, 0, 0, 0};
      if (steel_variant) {
        steel_scales.resize(scales.size());
        steel_biases.resize(scales.size());
        for (size_t i = 0; i < scales.size(); ++i) {
          steel_scales[i] = __fp16(scales[i]);
          steel_biases[i] = __fp16(-8.0f * scales[i]);
        }
        steel_scale_buffer = [device newBufferWithBytes:steel_scales.data() length:steel_scales.size() * sizeof(__fp16) options:MTLResourceStorageModeShared];
        steel_bias_buffer = [device newBufferWithBytes:steel_biases.data() length:steel_biases.size() * sizeof(__fp16) options:MTLResourceStorageModeShared];
        if (!steel_scale_buffer || !steel_bias_buffer) throw std::runtime_error("Metal allocation failed (steel scales/biases)");
        // Use private buffers and upload outside the measured interval.
        priv_act = [device newBufferWithLength:activation_buffer.length options:MTLResourceStorageModePrivate];
        priv_w = [device newBufferWithLength:packed_buffer.length options:MTLResourceStorageModePrivate];
        priv_sc = [device newBufferWithLength:steel_scale_buffer.length options:MTLResourceStorageModePrivate];
        priv_bi = [device newBufferWithLength:steel_bias_buffer.length options:MTLResourceStorageModePrivate];
        priv_out = [device newBufferWithLength:output_buffer.length options:MTLResourceStorageModePrivate];
      }

      // Upload private buffers once before timing.
      if (steel_variant && priv_w) {
        id<MTLCommandBuffer> up = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [up blitCommandEncoder];
        [blit copyFromBuffer:activation_buffer sourceOffset:0 toBuffer:priv_act destinationOffset:0 size:activation_buffer.length];
        [blit copyFromBuffer:packed_buffer sourceOffset:0 toBuffer:priv_w destinationOffset:0 size:packed_buffer.length];
        [blit copyFromBuffer:steel_scale_buffer sourceOffset:0 toBuffer:priv_sc destinationOffset:0 size:steel_scale_buffer.length];
        [blit copyFromBuffer:steel_bias_buffer sourceOffset:0 toBuffer:priv_bi destinationOffset:0 size:steel_bias_buffer.length];
        [blit endEncoding];
        [up commit];
        [up waitUntilCompleted];
      }
      std::vector<double> gpu_seconds;
      std::vector<double> wall_seconds;
      for (uint32_t iteration = 0; iteration < options.iterations + 1; ++iteration) {
        const auto wall_begin = std::chrono::steady_clock::now();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        if (steel_variant) {
          // Steel buffer bindings: 0=w, 1=scales, 2=biases, 3=x, 4=y.
          // Bindings 5–7 are K, N, M. Bind dummy batch metadata at 8–15.
          [encoder setBuffer:priv_w offset:0 atIndex:0];
          [encoder setBuffer:priv_sc offset:0 atIndex:1];
          [encoder setBuffer:priv_bi offset:0 atIndex:2];
          [encoder setBuffer:priv_act offset:0 atIndex:3];
          [encoder setBuffer:priv_out offset:0 atIndex:4];
          int k_int = int(options.k), n_int = int(options.n), m_int = int(options.m);
          [encoder setBytes:&k_int length:sizeof(k_int) atIndex:5];
          [encoder setBytes:&n_int length:sizeof(n_int) atIndex:6];
          [encoder setBytes:&m_int length:sizeof(m_int) atIndex:7];
          [encoder setBytes:&batch_ndims_zero length:sizeof(batch_ndims_zero) atIndex:8];
          [encoder setBytes:batch_shape_dummy length:sizeof(batch_shape_dummy) atIndex:9];
          [encoder setBytes:batch_strides_dummy length:sizeof(batch_strides_dummy) atIndex:10];
          [encoder setBytes:&batch_ndims_zero length:sizeof(batch_ndims_zero) atIndex:11];
          [encoder setBytes:batch_shape_dummy length:sizeof(batch_shape_dummy) atIndex:12];
          [encoder setBytes:batch_strides_dummy length:sizeof(batch_strides_dummy) atIndex:13];
          [encoder setBytes:batch_strides_dummy length:sizeof(batch_strides_dummy) atIndex:14];
          [encoder setBytes:batch_strides_dummy length:sizeof(batch_strides_dummy) atIndex:15];
          const MTLSize groups = MTLSizeMake((options.n + config.tile_n - 1) / config.tile_n, (options.m + config.tile_m - 1) / config.tile_m, 1);
          [encoder dispatchThreadgroups:groups threadsPerThreadgroup:config.threads];
          [encoder endEncoding];
        } else if (split_k > 0) {
          // Onyx split-k bindings: 0=activation, 1=packed, 2=scales, 4=m,
          // 5=n, 6=k, 7=group size, 8=fp32 partials, 9=split_k. The reduce
          // encoder joins the same command buffer so the GPU timestamp
          // spans both passes.
          [encoder setBuffer:activation_buffer offset:0 atIndex:0];
          [encoder setBuffer:packed_buffer offset:0 atIndex:1];
          [encoder setBuffer:scale_buffer offset:0 atIndex:2];
          [encoder setBytes:&options.m length:sizeof(options.m) atIndex:4];
          [encoder setBytes:&options.n length:sizeof(options.n) atIndex:5];
          [encoder setBytes:&options.k length:sizeof(options.k) atIndex:6];
          [encoder setBytes:&options.group length:sizeof(options.group) atIndex:7];
          [encoder setBuffer:partials_buffer offset:0 atIndex:8];
          [encoder setBytes:&split_k length:sizeof(split_k) atIndex:9];
          const MTLSize groups = MTLSizeMake((options.n + config.tile_n - 1) / config.tile_n, (options.m + config.tile_m - 1) / config.tile_m, split_k);
          [encoder dispatchThreadgroups:groups threadsPerThreadgroup:config.threads];
          [encoder endEncoding];
          id<MTLComputeCommandEncoder> reduce_encoder = [command computeCommandEncoder];
          [reduce_encoder setComputePipelineState:reduce_pipeline];
          [reduce_encoder setBuffer:partials_buffer offset:0 atIndex:0];
          [reduce_encoder setBuffer:output_buffer offset:0 atIndex:1];
          [reduce_encoder setBytes:&options.m length:sizeof(options.m) atIndex:2];
          [reduce_encoder setBytes:&options.n length:sizeof(options.n) atIndex:3];
          [reduce_encoder setBytes:&split_k length:sizeof(split_k) atIndex:4];
          const MTLSize reduce_groups = MTLSizeMake((options.m * options.n + 255) / 256, 1, 1);
          [reduce_encoder dispatchThreadgroups:reduce_groups threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [reduce_encoder endEncoding];
        } else {
          // Onyx bindings: 0=activation, 1=packed, 2=scales, 3=output, 4=m, 5=n, 6=k, 7=group size.
          [encoder setBuffer:activation_buffer offset:0 atIndex:0];
          [encoder setBuffer:packed_buffer offset:0 atIndex:1];
          [encoder setBuffer:scale_buffer offset:0 atIndex:2];
          [encoder setBuffer:output_buffer offset:0 atIndex:3];
          [encoder setBytes:&options.m length:sizeof(options.m) atIndex:4];
          [encoder setBytes:&options.n length:sizeof(options.n) atIndex:5];
          [encoder setBytes:&options.k length:sizeof(options.k) atIndex:6];
          [encoder setBytes:&options.group length:sizeof(options.group) atIndex:7];
          const MTLSize groups = MTLSizeMake((options.n + config.tile_n - 1) / config.tile_n, (options.m + config.tile_m - 1) / config.tile_m, 1);
          [encoder dispatchThreadgroups:groups threadsPerThreadgroup:config.threads];
          [encoder endEncoding];
        }
        [command commit];
        [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) {
          throw std::runtime_error("Metal command failed: " + std::string(command.error.localizedDescription.UTF8String));
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_begin).count();
        if (iteration > 0) {
          gpu_seconds.push_back(command.GPUEndTime - command.GPUStartTime);
          wall_seconds.push_back(wall);
        }
      }
      if (steel_variant && priv_out) {
        id<MTLCommandBuffer> down = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [down blitCommandEncoder];
        [blit copyFromBuffer:priv_out sourceOffset:0 toBuffer:output_buffer destinationOffset:0 size:output_buffer.length];
        [blit endEncoding];
        [down commit];
        [down waitUntilCompleted];
      }
      const __fp16* metal_output = static_cast<const __fp16*>(output_buffer.contents);
      double max_absolute = 0.0;
      double max_relative = 0.0;
      size_t checked = 0;
      bool correct = true;
      const size_t checks = std::min<size_t>(output_count, output_count <= 65536 ? output_count : 32);
      for (size_t check = 0; check < checks; ++check) {
        const size_t output_index = output_count <= 65536 ? check : (check * output_count / checks);
        const uint32_t row = output_index / options.n;
        const uint32_t column = output_index % options.n;
        float expected = 0.0f;
        for (uint32_t inner = 0; inner < options.k; ++inner) {
          const uint8_t byte = packed[column * (options.k / 2) + inner / 2];
          const int quantized = int((inner & 1) ? (byte >> 4) : (byte & 15)) - 8;
          const float weight = quantized * scales[column * (options.k / options.group) + inner / options.group];
          expected += half_to_float(activation[row * options.k + inner]) * weight;
        }
        const float actual = half_to_float(metal_output[output_index]);
        const double absolute = std::abs(actual - expected);
        const double relative = absolute / std::max(1e-6, static_cast<double>(std::abs(expected)));
        max_absolute = std::max(max_absolute, absolute);
        max_relative = std::max(max_relative, relative);
        correct = correct && absolute <= 0.05 + 0.005 * std::abs(expected);
        ++checked;
      }
      auto summarize = [](std::vector<double> values) {
        std::sort(values.begin(), values.end());
        const double median = values[values.size() / 2];
        const double p95 = values[std::min(values.size() - 1, size_t(std::ceil(values.size() * 0.95)) - 1)];
        return std::pair<double, double>(median, p95);
      };
      const auto [gpu_median, gpu_p95] = summarize(gpu_seconds);
      const auto [wall_median, wall_p95] = summarize(wall_seconds);
      const double flops = 2.0 * options.m * options.n * options.k;
      const double bytes = double(activation.size() * sizeof(__fp16) + packed.size() + scales.size() * sizeof(float) + output_count * sizeof(__fp16));
      char json[8192];
      std::snprintf(json, sizeof(json),
        "{\n"
        "  \"operator\": \"onyx.q4_gemm.v1\",\n"
        "  \"variant\": \"%s\",\n"
        "  \"device\": \"%s\",\n"
        "  \"shape\": {\"m\": %u, \"n\": %u, \"k\": %u, \"group_size\": %u},\n"
        "  \"correct\": %s,\n"
        "  \"checked_outputs\": %zu,\n"
        "  \"max_abs_error\": %.9g,\n"
        "  \"max_relative_error\": %.9g,\n"
        "  \"iterations\": %u,\n"
        "  \"gpu_median_ms\": %.6f,\n"
        "  \"gpu_p95_ms\": %.6f,\n"
        "  \"wall_median_ms\": %.6f,\n"
        "  \"wall_p95_ms\": %.6f,\n"
        "  \"effective_tflops\": %.6f,\n"
        "  \"effective_bandwidth_gb_per_s\": %.6f,\n"
        "  \"metal_allocated_bytes\": %llu,\n"
        "  \"recommended_working_set_bytes\": %llu,\n"
        "  \"max_buffer_bytes\": %llu\n"
        "}",
        options.variant.c_str(), device.name.UTF8String, options.m, options.n, options.k, options.group,
        correct ? "true" : "false", checked, max_absolute, max_relative,
        options.iterations, gpu_median * 1000, gpu_p95 * 1000,
        wall_median * 1000, wall_p95 * 1000, flops / gpu_median / 1e12,
        bytes / gpu_median / 1e9, static_cast<unsigned long long>(device.currentAllocatedSize),
        static_cast<unsigned long long>(device.recommendedMaxWorkingSetSize), static_cast<unsigned long long>(device.maxBufferLength));
      std::string result(json);
      std::ostringstream extensions;
      extensions << ",\n  \"gpu_samples_ms\": " << json_numbers(gpu_seconds, 1000.0)
                 << ",\n  \"wall_samples_ms\": " << json_numbers(wall_seconds, 1000.0)
                 << ",\n  \"arguments\": " << json_arguments(argc, argv)
                 << ",\n  \"metallib\": \"" << json_escape(ONYX_METALLIB_PATH) << "\"\n";
      result.insert(result.size() - 1, extensions.str());
      std::puts(result.c_str());
      write_json(options.output, result);
      return correct ? 0 : 2;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "onyx-metal-bench: %s\n", error.what());
      return 1;
    }
  }
}
