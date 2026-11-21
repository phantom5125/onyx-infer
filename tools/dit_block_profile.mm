#import "native.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

// Synthetic DiT-block profiler: drives the exact engine op sequence of one
// transformer block at production shapes (m=1024, hidden 5376, ffn 14336,
// 56x128 attention) through GraphSession, and reports the per-op wall-time
// table from session.profile_json(). Perf evidence only: random nibbles.
namespace {

constexpr uint32_t kHidden = 5376;
constexpr uint32_t kFfn = 14336;
constexpr uint32_t kRows = 1024;
constexpr uint32_t kHeadsDim = 128;

struct TensorSpec {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  size_t bytes;
  uint64_t begin;
};

}  // namespace

int main() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr, "no device\n"); return 1; }
    onyx::GraphSession session(device);
    if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);

    std::mt19937 rng(3);
    std::vector<TensorSpec> specs;
    size_t data_bytes = 0;
    auto add_q4 = [&](const std::string& name, uint32_t n, uint32_t k) {
      const size_t packed = size_t(n) * k / 2;
      const size_t scales = size_t(n) * (k / 64) * 4;
      specs.push_back({name, "Q4S64", {n, k}, packed + scales, data_bytes});
      data_bytes += packed + scales;
    };
    auto add_f16 = [&](const std::string& name, uint32_t count) {
      specs.push_back({name, "F16", {count}, size_t(count) * 2, data_bytes});
      data_bytes += size_t(count) * 2;
    };
    add_q4("b.attn.qkv_proj.weight", 3 * 56 * kHeadsDim, kHidden);
    add_q4("b.attn.out_proj.weight", kHidden, 56 * kHeadsDim);  // attention output is 7168-wide
    add_q4("b.mlp.fc1.weight", 2 * kFfn, kHidden);
    add_q4("b.mlp.fc2.weight", kHidden, kFfn);
    add_f16("b.attn.q_norm.weight", kHeadsDim);
    add_f16("b.attn.k_norm.weight", kHeadsDim);
    add_f16("b.norm1.weight", kHidden);
    add_f16("b.norm2.weight", kHidden);
    add_f16("mod.scale", kHidden);
    add_f16("mod.shift", kHidden);

    // Assemble the WeightStore file: [8B json length][json][tensor bytes].
    NSMutableDictionary* json_specs = [NSMutableDictionary dictionary];
    std::vector<uint8_t> data(data_bytes, 0);
    std::mt19937 fill(5);
    for (auto& spec : specs) {
      if (spec.dtype == "Q4S64") {
        for (size_t index = 0; index + 1 < spec.bytes; index += 2) {
          data[spec.begin + index] = uint8_t(fill() % 16);
          data[spec.begin + index + 1] = uint8_t(fill() % 16);
        }
        // scales region: positive floats
        const size_t scales_begin = spec.begin + (size_t(spec.shape[0]) * spec.shape[1] / 2);
        float* scale_data = reinterpret_cast<float*>(data.data() + scales_begin);
        for (size_t index = 0; index < size_t(spec.shape[1]) / 64 * spec.shape[0]; ++index) scale_data[index] = 0.01f;
      } else {
        __fp16* values = reinterpret_cast<__fp16*>(data.data() + spec.begin);
        for (size_t index = 0; index < spec.bytes / 2; ++index) values[index] = __fp16(0.5f + 0.01f * float(fill() % 8));
      }
      uint64_t packed_end = spec.begin + spec.bytes;
      uint64_t scale_begin = spec.begin, scale_end = spec.begin;
      if (spec.dtype == "Q4S64") {
        packed_end = spec.begin + size_t(spec.shape[0]) * spec.shape[1] / 2;
        scale_begin = packed_end;
        scale_end = spec.begin + spec.bytes;
      }
      NSMutableDictionary* entry = [NSMutableDictionary dictionary];
      entry[@"name"] = [NSString stringWithUTF8String:spec.name.c_str()];
      entry[@"dtype"] = [NSString stringWithUTF8String:spec.dtype.c_str()];
      entry[@"data_offsets"] = @[@(uint64_t(spec.begin)), @(packed_end)];
      entry[@"scale_offsets"] = @[@(scale_begin), @(scale_end)];
      json_specs[entry[@"name"]] = entry;
    }
    // rewrite shapes with real values (JSON arrays of numbers)
    for (size_t index = 0; index < specs.size(); ++index) {
      NSMutableArray* shape = [NSMutableArray array];
      for (int64_t dim : specs[index].shape) [shape addObject:@(dim)];
      json_specs[[NSString stringWithUTF8String:specs[index].name.c_str()]][@"shape"] = shape;
    }
    NSData* json = [NSJSONSerialization dataWithJSONObject:json_specs options:0 error:nil];
    const std::string store_path = std::string(NSTemporaryDirectory().UTF8String) + "onyx-dit-block-profile.bin";
    {
      std::ofstream out(store_path, std::ios::binary);
      const uint64_t length = json.length;
      out.write(reinterpret_cast<const char*>(&length), 8);
      out.write(static_cast<const char*>(json.bytes), json.length);
      out.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    onyx::WeightStore store(store_path, getenv("ONYX_RESIDENT") ? true : false);

    // Activations.
    std::vector<uint16_t> activation(size_t(kRows) * kHidden);
    {
      std::normal_distribution<float> normal(0.0f, 0.2f);
      for (auto& value : activation) {
        const __fp16 half = __fp16(normal(rng));
        value = *reinterpret_cast<const uint16_t*>(&half);
      }
    }
    auto activation_buffer = [device newBufferWithBytes:activation.data() length:activation.size() * 2 options:MTLResourceStorageModeShared];
    onyx::Tensor x{activation_buffer, {kRows, kHidden}, MPSDataTypeFloat16};

    // One block, matching dit.mm's sequence exactly (rms_norm + elementwise
    // modulation, qkv_attention, mlp with gated silu).
    auto run_block = [&]() {
      onyx::Tensor h = session.rms_norm(x, store.tensor("b.norm1.weight"), 1e-5);
      onyx::Tensor fused = session.linear(store, "b.attn.qkv_proj", h);
      onyx::Tensor attended = session.fused_qkv_attention(fused, store.tensor("b.attn.q_norm.weight"),
                                                          store.tensor("b.attn.k_norm.weight"), nullptr, nullptr);
      onyx::Tensor projected = session.linear(store, "b.attn.out_proj", attended);
      x = session.add(x, projected);
      h = session.rms_norm(x, store.tensor("b.norm2.weight"), 1e-5);
      onyx::Tensor hidden = session.linear(store, "b.mlp.fc1", h);
      onyx::Tensor gate = session.slice(hidden, 1, 0, hidden.shape[1] / 2);
      onyx::Tensor up = session.slice(hidden, 1, hidden.shape[1] / 2, hidden.shape[1] / 2);
      onyx::Tensor mlp_out = session.linear(store, "b.mlp.fc2", session.multiply(session.silu(gate), up));
      x = session.add(x, mlp_out);
    };

    for (int warm = 0; warm < 3; ++warm) run_block();
    const auto began = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < 10; ++iteration) run_block();
    const double block_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count() / 10.0;
    std::printf("block median-ish: %.2f ms | extrapolated over 50 blocks x 6 steps: %.1f s\n",
        block_ms, block_ms * 50 * 6 / 1000.0);
    std::printf("%s\n", session.profile_json().c_str());
    std::remove(store_path.c_str());
    return 0;
  }
}
