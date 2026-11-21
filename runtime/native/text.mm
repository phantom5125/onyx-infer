#import "native.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace onyx {
namespace {

Tensor bf16_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
  std::vector<uint16_t> encoded;
  encoded.reserve(values.size());
  for (float value : values) encoded.push_back(float_to_bf16(value));
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithBytes:encoded.data() length:encoded.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
  return Tensor{buffer, shape, MPSDataTypeBFloat16};
}

std::pair<Tensor, Tensor> text_rope_angles(uint64_t tokens) {
  std::vector<float> cosine(tokens * 128);
  std::vector<float> sine(tokens * 128);
  for (uint64_t position = 0; position < tokens; ++position) {
    for (uint64_t dimension = 0; dimension < 128; ++dimension) {
      const double frequency = std::pow(5000000.0, -static_cast<double>(dimension % 64) / 64.0);
      const double angle = position * frequency;
      const size_t index = position * 128 + dimension;
      cosine[index] = std::cos(angle);
      sine[index] = std::sin(angle);
    }
  }
  const std::vector<int64_t> shape = {1, 1, static_cast<int64_t>(tokens), 128};
  return {bf16_tensor(shape, cosine), bf16_tensor(shape, sine)};
}

}  // namespace

Tensor encode_text(const std::string& model, const std::string& tokens_path, const std::string& output, const StageOptions& options) {
  const auto began = std::chrono::steady_clock::now();
  require_streaming_weights(options, "text", "W4 text pack 21.86 GB exceeds the resident budget");
  const auto ids = read_token_ids(tokens_path);
  const uint64_t count = ids.size();
  WeightStore store(model);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  Tensor x = store.embedding("model.language_model.embed_tokens.weight", ids);
  auto [cosine, sine] = text_rope_angles(count);
  for (uint32_t layer = 0; layer < 50; ++layer) {
    @autoreleasepool {
      const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
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
      if ((layer + 1) % 10 == 0) std::fprintf(stderr, "native text layer %u/50\n", layer + 1);
    }
  }
  save_tensor(output, "embedding", x);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"text\",\"profile\":\"%s\",\"tokens\":%llu,\"seconds\":%.6f,\"output\":\"%s\"}\n",
              options.profile.c_str(), static_cast<unsigned long long>(count), seconds, output.c_str());
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
  }
  return x;
}

}  // namespace onyx
