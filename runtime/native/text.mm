#import "native.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace onyx {
namespace {

// Weights one encoder layer consumes, in the order the layer reads them.
const std::vector<std::string> kLayerTensors = {
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight",
    "post_attention_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};

Tensor bf16_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
  std::vector<uint16_t> encoded;
  encoded.reserve(values.size());
  for (float value : values) encoded.push_back(float_to_bf16(value));
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithBytes:encoded.data() length:encoded.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
  return Tensor{buffer, shape, MPSDataTypeBFloat16};
}

std::pair<Tensor, Tensor> text_rope_angles(uint64_t tokens, const int32_t* positions) {
  std::vector<float> cosine(tokens * 128);
  std::vector<float> sine(tokens * 128);
  for (uint64_t position = 0; position < tokens; ++position) {
    for (uint64_t dimension = 0; dimension < 128; ++dimension) {
      const uint64_t frequency_index = dimension % 64;
      const double frequency = std::pow(5000000.0, -static_cast<double>(frequency_index) / 64.0);
      // Qwen3-VL interleaves the height and width frequencies with time, leaving 24 for time.
      const uint64_t axis = frequency_index < 60 ? frequency_index % 3 : 0;
      const double coordinate = positions ? positions[position * 3 + axis] : position;
      const double angle = coordinate * frequency;
      const size_t index = position * 128 + dimension;
      cosine[index] = std::cos(angle);
      sine[index] = std::sin(angle);
    }
  }
  const std::vector<int64_t> shape = {1, 1, static_cast<int64_t>(tokens), 128};
  return {bf16_tensor(shape, cosine), bf16_tensor(shape, sine)};
}

void merge_visual_rows(Tensor& embedding, const Tensor& visual, const Tensor& rows, bool add) {
  const int64_t width = embedding.shape[1];
  if (rows.dtype != MPSDataTypeInt32 || rows.shape.size() != 1 ||
      visual.shape != std::vector<int64_t>{rows.shape[0], width} ||
      visual.dtype != MPSDataTypeBFloat16 || embedding.dtype != MPSDataTypeBFloat16) {
    throw std::runtime_error("Invalid H3 vision embedding layout");
  }
  const auto* indices = static_cast<const int32_t*>(rows.data());
  const auto* values = static_cast<const uint16_t*>(visual.data());
  auto* dst = static_cast<uint16_t*>(embedding.data());
  for (int64_t index = 0; index < rows.shape[0]; ++index) {
    const int32_t row = indices[index];
    if (row < 0 || row >= embedding.shape[0] || (index > 0 && row <= indices[index - 1])) {
      throw std::runtime_error("Invalid H3 vision token index");
    }
    auto* output = dst + row * width;
    const auto* input = values + index * width;
    if (add) {
      for (int64_t column = 0; column < width; ++column) {
        output[column] = float_to_bf16(bf16_to_float(output[column]) + bf16_to_float(input[column]));
      }
    } else {
      std::memcpy(output, input, width * sizeof(uint16_t));
    }
  }
}

}  // namespace

Tensor encode_text(const std::string& model, const std::string& tokens_path, const std::string& output, const StageOptions& options, const std::string& vision_path) {
  const auto began = std::chrono::steady_clock::now();
  require_streaming_weights(options, "text", "W4 text pack 21.86 GB exceeds the resident budget");
  const auto ids = read_token_ids(tokens_path);
  const uint64_t count = ids.size();
  WeightStore store(model);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  Tensor x = store.embedding("model.language_model.embed_tokens.weight", ids);
  std::unique_ptr<WeightStore> vision;
  Tensor positions{};
  Tensor visual_rows{};
  if (!vision_path.empty()) {
    vision = std::make_unique<WeightStore>(vision_path);
    positions = vision->tensor("positions");
    if (positions.dtype != MPSDataTypeInt32 ||
        positions.shape != std::vector<int64_t>{static_cast<int64_t>(count), 3}) {
      throw std::runtime_error("Invalid H3 vision RoPE positions");
    }
    visual_rows = vision->tensor("rows");
    merge_visual_rows(x, vision->tensor("vision"), visual_rows, false);
  }
  auto [cosine, sine] = text_rope_angles(count, vision ? static_cast<const int32_t*>(positions.data()) : nullptr);
  for (uint32_t layer = 0; layer < 50; ++layer) {
    @autoreleasepool {
      const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
      // The next layer's weights stream in while this layer computes.
      if (options.prefetch_weights && layer + 1 < 50) {
        store.prefetch("model.language_model.layers." + std::to_string(layer + 1) + ".", kLayerTensors);
      }
      Tensor h = session.rms_norm(x, store.tensor(prefix + "input_layernorm.weight"), 1e-6);
      const std::string attention = prefix + "self_attn.";
      Tensor q = session.linear(store, attention + "q_proj", h);
      Tensor k = session.linear(store, attention + "k_proj", h);
      Tensor v = session.linear(store, attention + "v_proj", h);
      q = session.reshape(q, {1, static_cast<int64_t>(count), 64, 128});
      k = session.reshape(k, {1, static_cast<int64_t>(count), 8, 128});
      v = session.reshape(v, {1, static_cast<int64_t>(count), 8, 128});
      q = session.transpose(q, {0, 2, 1, 3});
      k = session.transpose(k, {0, 2, 1, 3});
      v = session.transpose(v, {0, 2, 1, 3});
      q = session.rms_norm(q, store.tensor(attention + "q_norm.weight"), 1e-6);
      k = session.rms_norm(k, store.tensor(attention + "k_norm.weight"), 1e-6);
      q = session.rope(q, cosine, sine);
      k = session.rope(k, cosine, sine);
      Tensor attended = session.attention(q, k, v, true, 1.0 / std::sqrt(128.0));
      attended = session.transpose(attended, {0, 2, 1, 3});
      attended = session.reshape(attended, {static_cast<int64_t>(count), 8192});
      attended = session.linear(store, attention + "o_proj", attended);
      x = session.add(x, attended);
      h = session.rms_norm(x, store.tensor(prefix + "post_attention_layernorm.weight"), 1e-6);
      Tensor gate = session.linear(store, prefix + "mlp.gate_proj", h);
      Tensor up = session.linear(store, prefix + "mlp.up_proj", h);
      Tensor hidden = session.multiply(session.silu(gate), up);
      hidden = session.linear(store, prefix + "mlp.down_proj", hidden);
      x = session.add(x, hidden);
      if (vision && layer < 3) {
        merge_visual_rows(x, vision->tensor("deepstack_" + std::to_string(layer)), visual_rows, true);
      }
      if ((layer + 1) % 10 == 0) std::fprintf(stderr, "native text layer %u/50\n", layer + 1);
    }
  }
  if (options.prefetch_weights) store.finish_prefetch();
  save_tensor(output, "embedding", x);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"text\",\"profile\":\"%s\",\"tokens\":%llu,\"seconds\":%.6f,\"output\":\"%s\","
              "\"vision_rows\":%lld,\"prefetch_weights\":%s,\"prefetch_wait_seconds\":%.6f}\n",
              options.profile.c_str(), static_cast<unsigned long long>(count), seconds, output.c_str(),
              vision ? static_cast<long long>(visual_rows.shape[0]) : 0LL,
              options.prefetch_weights ? "true" : "false", store.prefetch_wait_seconds());
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
  }
  return x;
}

}  // namespace onyx
