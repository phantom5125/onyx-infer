#import "native.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace onyx {
namespace {

constexpr int64_t kLatentChannels = 24;
constexpr int64_t kTileLatents = 16;
constexpr int64_t kPixelsPerLatent = 16;
constexpr int64_t kPatchTokens = kTileLatents * kTileLatents;
constexpr int64_t kRegisterTokens = 4;
constexpr int64_t kSuffixTokens = kRegisterTokens + 1;  // Register tokens followed by a null token.
constexpr int64_t kSequenceTokens = kPatchTokens + kSuffixTokens;
constexpr int64_t kTilePixels = kTileLatents * kPixelsPerLatent;
constexpr size_t kTileValues = kTilePixels * kTilePixels * 3;

Tensor float_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithBytes:values.data() length:values.size() * sizeof(float) options:MTLResourceStorageModeShared];
  if (!buffer) throw std::runtime_error("Could not allocate VAE constants");
  return Tensor{buffer, shape, MPSDataTypeFloat32};
}

std::vector<int32_t> tile_positions(int32_t size) {
  constexpr int32_t overlap = 4;
  constexpr int32_t stride = kTileLatents - overlap;
  if (size <= kTileLatents) return {0};
  const int32_t count = (size - overlap + stride - 1) / stride;
  const int32_t extra = stride * count + overlap - size;
  std::vector<int32_t> result = {0};
  int32_t current = 0;
  for (int32_t index = 0; index < count - 1; ++index) {
    current += stride - extra / (count - 1) - static_cast<int32_t>(index < extra % (count - 1));
    result.push_back(current);
  }
  return result;
}

std::pair<Tensor, Tensor> vae_cosine_sine() {
  std::vector<float> cosine(kSequenceTokens * 32, 1.0f);
  std::vector<float> sine(kSequenceTokens * 32, 0.0f);
  for (uint32_t y = 0; y < kTileLatents; ++y) {
    for (uint32_t x = 0; x < kTileLatents; ++x) {
      const double position[] = {(y + 0.5) * 2 / kTileLatents - 1, (x + 0.5) * 2 / kTileLatents - 1};
      const uint32_t token = y * kTileLatents + x;
      for (uint32_t axis = 0; axis < 2; ++axis) {
        for (uint32_t index = 0; index < 8; ++index) {
          const double angle = 2 * M_PI * position[axis] * std::pow(100.0, -static_cast<double>(index) / 8.0);
          for (uint32_t copy = 0; copy < 2; ++copy) {
            const uint32_t output = token * 32 + copy * 16 + axis * 8 + index;
            cosine[output] = std::cos(angle);
            sine[output] = std::sin(angle);
          }
        }
      }
    }
  }
  return {float_tensor({1, 1, kSequenceTokens, 32}, cosine), float_tensor({1, 1, kSequenceTokens, 32}, sine)};
}

size_t decode_batch() {
  static const size_t count = [] {
    const char* value = std::getenv("ONYX_VAE_BATCH");
    const long long parsed = value ? std::atoll(value) : 25;
    return parsed > 0 ? static_cast<size_t>(parsed) : 1;
  }();
  return count;
}

Tensor tile_tokens(GraphSession& session, WeightStore& store, int64_t batch, int64_t dim) {
  const Tensor weight = session.reshape(store.tensor("decoder.register_tokens"), {kRegisterTokens, dim});
  const Tensor registers = session.cast(weight, MPSDataTypeFloat32);
  const auto* source = static_cast<const float*>(registers.data());
  std::vector<float> values(batch * kSuffixTokens * dim, 0.0f);
  for (int64_t tile = 0; tile < batch; ++tile) {
    std::copy_n(source, kRegisterTokens * dim, values.data() + tile * kSuffixTokens * dim);
  }
  return float_tensor({batch, kSuffixTokens, dim}, values);
}

Tensor pack_tiles(id<MTLDevice> device, const Tensor& latent, const std::vector<int32_t>& ys,
                  const std::vector<int32_t>& xs, size_t first, int64_t count) {
  Tensor result{nil, {count, kPatchTokens, kLatentChannels}, MPSDataTypeFloat32};
  result.buffer = [device newBufferWithLength:result.byte_size() options:MTLResourceStorageModeShared];
  if (!result.buffer) throw std::runtime_error("Could not allocate VAE tile input");
  const auto* source = static_cast<const float*>(latent.data());
  auto* destination = static_cast<float*>(result.data());
  const int64_t width = latent.shape[1];
  constexpr int64_t row_values = kTileLatents * kLatentChannels;
  for (int64_t tile = 0; tile < count; ++tile) {
    const size_t index = first + tile;
    const int64_t y = ys[index / xs.size()];
    const int64_t x = xs[index % xs.size()];
    for (int64_t row = 0; row < kTileLatents; ++row) {
      std::copy_n(source + ((y + row) * width + x) * kLatentChannels, row_values, destination);
      destination += row_values;
    }
  }
  return result;
}

Tensor decode_tiles(GraphSession& session, WeightStore& store, const Tensor& z, const Tensor& tokens,
                    const Tensor& ones, const Tensor& cosine, const Tensor& sine) {
  const int64_t batch = z.shape[0];
  const bool fused = batch > 1;
  Tensor post = session.reshape(store.tensor("post_quant_conv.weight"), {kLatentChannels, kLatentChannels});
  Tensor x = session.matmul(z, post);
  x = session.add(x, store.tensor("post_quant_conv.bias"));
  x = session.linear(store, "decoder.x_embedder", x);
  const int64_t dim = tokens.shape.back();
  x = session.concat({x, tokens}, 1);
  for (uint32_t layer = 0; layer < 24; ++layer) {
    const std::string prefix = "decoder.transformer_blocks." + std::to_string(layer);
    Tensor h = session.rms_norm(x, store.tensor(prefix + ".norm1.weight"), 1e-5);
    Tensor qkv = session.linear(store, prefix + ".attn.to_qkv", h);
    qkv = session.reshape(qkv, {batch, kSequenceTokens, dim / 64, 192});
    Tensor attended;
    if (fused) {
      attended = session.tile_attention(qkv, cosine, sine);
    } else {
      qkv = session.transpose(qkv, {0, 2, 1, 3});
      Tensor parts[3];
      for (uint32_t index = 0; index < 3; ++index) parts[index] = session.slice(qkv, 3, index * 64, 64);
      parts[0] = session.rms_norm(parts[0], ones, 1e-5);
      parts[1] = session.rms_norm(parts[1], ones, 1e-5);
      parts[0] = session.rope(parts[0], cosine, sine);
      parts[1] = session.rope(parts[1], cosine, sine);
      attended = session.attention(parts[0], parts[1], parts[2], false, 0.125);
      attended = session.reshape(session.transpose(attended, {0, 2, 1, 3}), {batch, kSequenceTokens, dim});
    }
    attended = session.linear(store, prefix + ".attn.to_out", attended);
    Tensor scale = store.tensor(prefix + ".scale1");
    x = fused ? session.gated_residual(x, attended, scale) : session.add(x, session.multiply(attended, scale));
    h = session.rms_norm(x, store.tensor(prefix + ".norm2.weight"), 1e-5);
    Tensor hidden = session.linear(store, prefix + ".ff.w1", h);
    if (fused) {
      hidden = session.gated_activation(hidden);
    } else {
      Tensor gate = session.slice(hidden, 2, 0, hidden.shape[2] / 2);
      Tensor up = session.slice(hidden, 2, hidden.shape[2] / 2, hidden.shape[2] / 2);
      hidden = session.multiply(session.silu(gate), up);
    }
    hidden = session.linear(store, prefix + ".ff.w2", hidden);
    scale = store.tensor(prefix + ".scale2");
    x = fused ? session.gated_residual(x, hidden, scale) : session.add(x, session.multiply(hidden, scale));
  }
  x = session.slice(x, 1, 0, kPatchTokens);
  x = session.layer_norm(x, store.tensor("decoder.norm_out.weight"), store.tensor("decoder.norm_out.bias"), 1e-5);
  x = session.linear(store, "decoder.proj_out", x);
  x = session.reshape(x, {batch, kTileLatents, kTileLatents, 3, kPixelsPerLatent, kPixelsPerLatent});
  x = session.transpose(x, {0, 1, 4, 2, 5, 3});
  return session.reshape(x, {batch, kTilePixels, kTilePixels, 3});
}

void blend_tile(const float* tile, const std::vector<float>& previous, std::vector<float>& strip,
                size_t width, size_t x, size_t vertical, size_t horizontal) {
  const size_t stride = width * 3;
  for (size_t py = 0; py < kTilePixels; ++py) {
    const float* source = tile + py * kTilePixels * 3;
    float* target = strip.data() + py * stride + x * 3;
    if (py < vertical) {
      const float* above = previous.data() + (kTilePixels - vertical + py) * stride + x * 3;
      const float alpha = static_cast<float>(py) / vertical;
      for (size_t i = 0; i < horizontal * 3; ++i) {
        const float value = above[i] + (source[i] - above[i]) * alpha;
        const float left_alpha = static_cast<float>(i / 3) / horizontal;
        target[i] += (value - target[i]) * left_alpha;
      }
      for (size_t i = horizontal * 3; i < kTilePixels * 3; ++i) {
        target[i] = above[i] + (source[i] - above[i]) * alpha;
      }
    } else {
      for (size_t i = 0; i < horizontal * 3; ++i) {
        const float alpha = static_cast<float>(i / 3) / horizontal;
        target[i] += (source[i] - target[i]) * alpha;
      }
      std::copy(source + horizontal * 3, source + kTilePixels * 3, target + horizontal * 3);
    }
  }
}

void finalize_rows(const std::vector<float>& strip, uint8_t* rgb, size_t values) {
  constexpr float mean[] = {0.485f, 0.456f, 0.406f};
  constexpr float variance[] = {0.229f, 0.224f, 0.225f};
  for (size_t index = 0; index < values; ++index) {
    const size_t channel = index % 3;
    const float value = strip[index] * variance[channel] + mean[channel];
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite VAE output");
    rgb[index] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
  }
}

void write_png(const std::string& path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgb) {
  @autoreleasepool {
    NSData* data = [NSData dataWithBytes:rgb.data() length:rgb.size()];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGImageRef image = CGImageCreate(width, height, 8, 24, width * 3, color_space,
                                     kCGImageAlphaNone, provider, nullptr, false, kCGRenderingIntentDefault);
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
        (__bridge CFURLRef)[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]], (__bridge CFStringRef)UTTypePNG.identifier, 1, nullptr);
    if (!destination) throw std::runtime_error("Could not create PNG destination");
    CGImageDestinationAddImage(destination, image, nullptr);
    const bool ok = CGImageDestinationFinalize(destination);
    CGImageRelease(image);
    CGColorSpaceRelease(color_space);
    CGDataProviderRelease(provider);
    CFRelease(destination);
    if (!ok) throw std::runtime_error("Could not write PNG");
  }
}

}  // namespace

Tensor decode_image(const std::string& model, const std::string& latent_path, const std::string& config_path, const std::string& output, const StageOptions& options) {
  const auto began = std::chrono::steady_clock::now();
  const size_t batch_limit = decode_batch();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  WeightStore store(model, options.weights == "resident");
  Tensor latent = load_tensor(latent_path, "latent");
  if (latent.shape.size() != 3 || latent.shape[0] != kLatentChannels ||
      latent.shape[1] < kTileLatents || latent.shape[2] < kTileLatents) {
    throw std::runtime_error("VAE latent must have shape [24, H, W] with H and W at least 16");
  }
  if (latent.dtype != MPSDataTypeFloat32) throw std::runtime_error("VAE latent must be float32");
  @autoreleasepool {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:config_path.c_str()]];
    if (!data) throw std::runtime_error("Missing VAE config");
    NSDictionary* config = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    std::vector<float> mean;
    std::vector<float> variance;
    for (NSNumber* value in config[@"latents_mean"]) mean.push_back(value.floatValue);
    for (NSNumber* value in config[@"latents_std"]) variance.push_back(value.floatValue);
    if (mean.size() != kLatentChannels || variance.size() != kLatentChannels) {
      throw std::runtime_error("VAE config must provide 24 latent means and standard deviations");
    }
    Tensor mean_tensor = float_tensor({kLatentChannels}, mean);
    Tensor std_tensor = float_tensor({kLatentChannels}, variance);
    latent = session.transpose(latent, {1, 2, 0});
    latent = session.add(session.multiply(latent, std_tensor), mean_tensor);
  }
  const uint32_t latent_height = latent.shape[0];
  const uint32_t latent_width = latent.shape[1];
  const auto ys = tile_positions(latent_height);
  const auto xs = tile_positions(latent_width);
  const uint32_t width = latent_width * kPixelsPerLatent;
  const uint32_t height = latent_height * kPixelsPerLatent;
  const size_t row_values = static_cast<size_t>(width) * 3;
  const size_t tile_count = ys.size() * xs.size();
  const size_t batch = std::min(batch_limit, tile_count);
  std::vector<uint8_t> rgb(row_values * height);
  std::vector<float> strip(kTilePixels * row_values);
  std::vector<float> previous(ys.size() > 1 ? strip.size() : 0);
  Tensor ones = float_tensor({64}, std::vector<float>(64, 1.0f));
  auto [cosine, sine] = vae_cosine_sine();
  const Tensor tokens = tile_tokens(session, store, batch, store.shape("decoder.x_embedder.weight").at(0));
  size_t finalized_y = 0;
  for (size_t first = 0; first < tile_count; first += batch) {
    @autoreleasepool {
      const size_t count = std::min(batch, tile_count - first);
      const Tensor packed = pack_tiles(device, latent, ys, xs, first, count);
      const Tensor decoded = decode_tiles(session, store, packed, session.slice(tokens, 0, 0, count),
                                          ones, cosine, sine);
      const auto* values = static_cast<const float*>(decoded.data());
      for (size_t index = 0; index < count; ++index) {
        const size_t row = (first + index) / xs.size();
        const size_t column = (first + index) % xs.size();
        const size_t y = ys[row] * kPixelsPerLatent;
        const size_t x = xs[column] * kPixelsPerLatent;
        if (column == 0 && row > 0) {
          // Later tiles start at y or below, so the rows above y can become RGB8.
          finalize_rows(strip, rgb.data() + finalized_y * row_values, (y - finalized_y) * row_values);
          strip.swap(previous);
          finalized_y = y;
        }
        const size_t vertical = row ? (ys[row - 1] + kTileLatents - ys[row]) * kPixelsPerLatent : 0;
        const size_t horizontal = column ? (xs[column - 1] + kTileLatents - xs[column]) * kPixelsPerLatent : 0;
        blend_tile(values + index * kTileValues, previous, strip, width, x, vertical, horizontal);
      }
    }
  }
  finalize_rows(strip, rgb.data() + finalized_y * row_values, (height - finalized_y) * row_values);
  write_png(output, width, height, rgb);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"vae\",\"profile\":\"%s\",\"weights\":\"%s\",\"seconds\":%.6f,\"output\":\"%s\","
              "\"tiles\":%zu,\"decode_batch\":%zu}\n",
              options.profile.c_str(), options.weights.c_str(), seconds, output.c_str(), tile_count, batch);
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
  }
  return latent;
}

}  // namespace onyx
