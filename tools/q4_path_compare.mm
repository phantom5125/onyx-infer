#import "native.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// Compare the Q4 linear paths on real packed weights through the engine entry point.
//
//   onyx-q4-path-compare <pack> <layer prefix> [rows]
//
// For MPSGraph with the FP32 dequantization, MPSGraph with the transpose moved off the
// weight (ONYX_Q4_GRAPH=f32-t), and the neural-accelerator kernel (ONYX_Q4_LINEAR=nax), it
// prints the median call time, whether the outputs match the first path bit for bit, and
// the error against FP64 reference products on a sample of outputs.
namespace {

struct Config {
  const char* mode;
  const char* graph;
};

double median_milliseconds(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) {
      std::fprintf(stderr, "usage: onyx-q4-path-compare <pack> <layer prefix> [rows]\n");
      return 2;
    }
    const std::string pack = argv[1];
    const std::string prefix = argv[2];
    const uint32_t rows = argc > 3 ? uint32_t(std::atoi(argv[3])) : 1093;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    onyx::GraphSession session(device);
    onyx::WeightStore store(pack, false);
    const uint32_t n = uint32_t(store.shape(prefix + ".weight")[0]);
    const uint32_t k = uint32_t(store.shape(prefix + ".weight")[1]);

    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<uint16_t> activation(rows * k);
    std::vector<double> activation_value(rows * k);
    for (size_t index = 0; index < activation.size(); ++index) {
      activation[index] = onyx::float_to_bf16(normal(rng));
      activation_value[index] = onyx::bf16_to_float(activation[index]);
    }
    auto buffer = [device newBufferWithBytes:activation.data() length:activation.size() * 2 options:MTLResourceStorageModeShared];
    const onyx::Tensor x{buffer, {rows, k}, MPSDataTypeBFloat16};

    const Config configs[] = {{"mpsgraph", ""}, {"mpsgraph", "f32-t"}, {"nax", ""}};
    std::vector<std::vector<uint16_t>> outputs;
    std::vector<double> medians;
    for (const Config& config : configs) {
      setenv("ONYX_Q4_LINEAR", config.mode, 1);
      setenv("ONYX_Q4_GRAPH", config.graph, 1);
      outputs.emplace_back(rows * n);
      std::copy_n(static_cast<const uint16_t*>(session.linear(store, prefix, x).buffer.contents), size_t(rows) * n, outputs.back().data());
      std::vector<double> samples;
      for (int call = 0; call < 12; ++call) {
        const auto begin = std::chrono::steady_clock::now();
        onyx::Tensor again = session.linear(store, prefix, x);
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
      }
      medians.push_back(median_milliseconds(samples));
    }
    unsetenv("ONYX_Q4_GRAPH");

    std::printf("rows %u n %u k %u, median of 12 calls through GraphSession::linear\n", rows, n, k);
    for (size_t index = 0; index < 3; ++index) {
      std::printf("  %-8s %-5s %8.3f ms\n", configs[index].mode, configs[index].graph, medians[index]);
    }
    std::printf("  transpose off the weight versus FP32 dequantization: %s\n",
                outputs[0] == outputs[1] ? "bit-identical" : "DIFFERENT");
    size_t differing = 0;
    double max_absolute = 0.0;
    for (size_t index = 0; index < outputs[0].size(); ++index) {
      if (outputs[0][index] == outputs[2][index]) continue;
      ++differing;
      const double a = onyx::bf16_to_float(outputs[0][index]);
      const double b = onyx::bf16_to_float(outputs[2][index]);
      max_absolute = std::max(max_absolute, std::abs(a - b));
    }
    std::printf("  neural accelerator versus MPSGraph: %zu of %zu outputs differ (%.2f%%), max abs %.4g\n",
                differing, outputs[0].size(), 100.0 * differing / outputs[0].size(), max_absolute);

    // FP64 reference products for a sample of outputs. `exact` uses FP32 scales. `affine`
    // uses the scale rounded to BF16 and the matching bias, before the product is rounded.
    const onyx::Tensor weight = store.tensor(prefix + ".weight");
    const onyx::Tensor scales = store.scale(prefix + ".weight");
    const uint8_t* packed = static_cast<const uint8_t*>(weight.data());
    const float* scale_values = static_cast<const float*>(scales.data());
    std::uniform_int_distribution<size_t> pick(0, size_t(rows) * n - 1);
    // Rounding the weights to the activation type leaves an absolute error of about the
    // same size on every output, so the fair unit is the half-ulp of the BF16 grid at the
    // typical output magnitude rather than at each output's own value.
    double worst[2][2] = {};
    double exact_square_sum = 0.0;
    std::vector<double> exact_values;
    std::vector<std::array<double, 2>> got_values;
    for (int check = 0; check < 96; ++check) {
      const size_t flat = pick(rng);
      const size_t row = flat / n, column = flat % n;
      double exact = 0.0, affine = 0.0;
      for (uint32_t inner = 0; inner < k; ++inner) {
        const uint8_t byte = packed[(size_t(column) * k + inner) / 2];
        const int nibble = inner & 1 ? byte >> 4 : byte & 15;
        const double scale = scale_values[size_t(column) * (k / 64) + inner / 64];
        const double scale_bf16 = onyx::bf16_to_float(onyx::float_to_bf16(float(scale)));
        const double bias_bf16 = onyx::bf16_to_float(onyx::float_to_bf16(float(-8.0 * scale)));
        exact += activation_value[size_t(row) * k + inner] * (nibble - 8) * scale;
        affine += activation_value[size_t(row) * k + inner] * (nibble * scale_bf16 + bias_bf16);
      }
      const double got[2] = {onyx::bf16_to_float(outputs[0][flat]), onyx::bf16_to_float(outputs[2][flat])};
      for (const int path : {0, 1}) {
        worst[path][0] = std::max(worst[path][0], std::abs(got[path] - exact));
        worst[path][1] = std::max(worst[path][1], std::abs(got[path] - affine));
      }
      exact_square_sum += exact * exact;
      exact_values.push_back(exact);
      got_values.push_back({got[0], got[1]});
    }
    const double typical = std::sqrt(exact_square_sum / exact_values.size());
    const double half_ulp = std::pow(2.0, std::floor(std::log2(typical)) - 8);
    double worst_typical[2] = {};
    for (size_t index = 0; index < exact_values.size(); ++index) {
      for (const int path : {0, 1}) {
        worst_typical[path] = std::max(worst_typical[path], std::abs(got_values[index][path] - exact_values[index]) / half_ulp);
      }
    }
    std::printf("max |output - reference| over 96 outputs, before the BF16 rounding of the output:\n");
    std::printf("  against FP32-scale weights:                  MPSGraph %.4g  neural accelerator %.4g\n", worst[0][0], worst[1][0]);
    std::printf("  against BF16 scale and bias, unrounded product: MPSGraph %.4g  neural accelerator %.4g\n", worst[0][1], worst[1][1]);
    std::printf("RMS of the exact outputs %.4g, half-ulp of the BF16 grid there %.4g. Max error in those half-ulps:\n", typical, half_ulp);
    std::printf("  MPSGraph %.3f  neural accelerator %.3f\n", worst_typical[0], worst_typical[1]);
    return 0;
  }
}
