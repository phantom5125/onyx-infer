#import "native.hpp"

#import <CoreGraphics/CoreGraphics.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>

namespace onyx {
namespace {

Tensor float_tensor(const std::vector<int64_t>& shape, const std::vector<float>& values) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithBytes:values.data() length:values.size() * sizeof(float) options:MTLResourceStorageModeShared];
  return Tensor{buffer, shape, MPSDataTypeFloat32};
}

std::vector<int32_t> tile_positions(int32_t size) {
  if (size <= 16) return {0};
  const int32_t count = std::ceil(static_cast<double>(size - 4) / 12.0);
  const int32_t extra = 16 * count - 4 * (count - 1) - size;
  std::vector<int32_t> result = {0};
  int32_t current = 0;
  for (int32_t index = 0; index < count - 1; ++index) {
    current += 12 - extra / (count - 1) - static_cast<int32_t>(index < extra % (count - 1));
    result.push_back(current);
  }
  return result;
}

std::pair<Tensor, Tensor> vae_cosine_sine() {
  std::vector<float> cosine(261 * 32);
  std::vector<float> sine(261 * 32);
  for (uint32_t y = 0; y < 16; ++y) {
    for (uint32_t x = 0; x < 16; ++x) {
      const double position[] = {(y + 0.5) * 2 / 16 - 1, (x + 0.5) * 2 / 16 - 1};
      const uint32_t token = y * 16 + x;
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
  for (uint32_t token = 256; token < 261; ++token) {
    std::fill(cosine.begin() + token * 32, cosine.begin() + (token + 1) * 32, 1.0f);
    std::fill(sine.begin() + token * 32, sine.begin() + (token + 1) * 32, 0.0f);
  }
  return {float_tensor({1, 1, 261, 32}, cosine), float_tensor({1, 1, 261, 32}, sine)};
}

// Tiles decoded in one pass. One 1024×1024 image has 25 tiles and 4096×4096 has 400.
// ONYX_VAE_BATCH overrides the count, and 1 decodes one tile at a time.
size_t decode_batch() {
  static const size_t count = [] {
    const char* value = std::getenv("ONYX_VAE_BATCH");
    const long long parsed = value ? std::atoll(value) : 25;
    return parsed > 0 ? static_cast<size_t>(parsed) : 1;
  }();
  return count;
}

Tensor decode_tile(GraphSession& session, WeightStore& store, const Tensor& z, const Tensor& ones, const Tensor& cosine, const Tensor& sine) {
  const uint64_t batch = z.shape[0];
  Tensor x = session.reshape(z, {static_cast<int64_t>(batch), 256, 24});
  Tensor post = session.reshape(store.tensor("post_quant_conv.weight"), {24, 24});
  x = session.matmul(x, post);
  x = session.add(x, store.tensor("post_quant_conv.bias"));
  x = session.linear(store, "decoder.x_embedder", x);
  const int64_t dim = store.shape("decoder.x_embedder.weight")[0];
  // Every tile in the batch carries the same registers and null token.
  std::vector<float> registers(static_cast<size_t>(batch) * 4 * static_cast<size_t>(dim));
  // The streaming store can release this buffer as soon as its Tensor is destroyed.
  id<MTLBuffer> register_buffer __attribute__((objc_precise_lifetime)) = store.tensor("decoder.register_tokens").buffer;
  const float* register_source = static_cast<const float*>(register_buffer.contents);
  for (size_t index = 0; index < registers.size(); ++index) registers[index] = register_source[index % (4 * static_cast<size_t>(dim))];
  Tensor null = float_tensor({static_cast<int64_t>(batch), 1, dim}, std::vector<float>(static_cast<size_t>(batch) * dim, 0.0f));
  x = session.concat({x, float_tensor({static_cast<int64_t>(batch), 4, dim}, registers), null}, 1);
  for (uint32_t layer = 0; layer < 24; ++layer) {
    const std::string prefix = "decoder.transformer_blocks." + std::to_string(layer);
    Tensor h = session.rms_norm(x, store.tensor(prefix + ".norm1.weight"), 1e-5);
    Tensor qkv = session.linear(store, prefix + ".attn.to_qkv", h);
    qkv = session.reshape(qkv, {static_cast<int64_t>(batch), 261, dim / 64, 192});
    qkv = session.transpose(qkv, {0, 2, 1, 3});
    Tensor parts[3];
    for (uint32_t index = 0; index < 3; ++index) parts[index] = session.slice(qkv, 3, index * 64, 64);
    parts[0] = session.rms_norm(parts[0], ones, 1e-5);
    parts[1] = session.rms_norm(parts[1], ones, 1e-5);
    parts[0] = session.rope(parts[0], cosine, sine);
    parts[1] = session.rope(parts[1], cosine, sine);
    Tensor attended = session.attention(parts[0], parts[1], parts[2], false, 0.125);
    attended = session.transpose(attended, {0, 2, 1, 3});
    attended = session.reshape(attended, {static_cast<int64_t>(batch), 261, dim});
    attended = session.linear(store, prefix + ".attn.to_out", attended);
    attended = session.multiply(attended, store.tensor(prefix + ".scale1"));
    x = session.add(x, attended);
    h = session.rms_norm(x, store.tensor(prefix + ".norm2.weight"), 1e-5);
    Tensor hidden = session.linear(store, prefix + ".ff.w1", h);
    Tensor gate = session.slice(hidden, 2, 0, hidden.shape[2] / 2);
    Tensor up = session.slice(hidden, 2, hidden.shape[2] / 2, hidden.shape[2] / 2);
    hidden = session.linear(store, prefix + ".ff.w2", session.multiply(session.silu(gate), up));
    hidden = session.multiply(hidden, store.tensor(prefix + ".scale2"));
    x = session.add(x, hidden);
  }
  x = session.slice(x, 1, 0, 256);
  x = session.layer_norm(x, store.tensor("decoder.norm_out.weight"), store.tensor("decoder.norm_out.bias"), 1e-5);
  x = session.linear(store, "decoder.proj_out", x);
  x = session.reshape(x, {static_cast<int64_t>(batch), 16, 16, 3, 16, 16});
  x = session.transpose(x, {0, 1, 4, 2, 5, 3});
  return session.reshape(x, {static_cast<int64_t>(batch), 256, 256, 3});
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
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  WeightStore store(model, options.weights == "resident");
  Tensor latent = load_tensor(latent_path, "latent");
  @autoreleasepool {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:config_path.c_str()]];
    if (!data) throw std::runtime_error("Missing VAE config");
    NSDictionary* config = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    std::vector<float> mean;
    std::vector<float> variance;
    for (NSNumber* value in config[@"latents_mean"]) mean.push_back(value.floatValue);
    for (NSNumber* value in config[@"latents_std"]) variance.push_back(value.floatValue);
    Tensor mean_tensor = float_tensor({24}, mean);
    Tensor std_tensor = float_tensor({24}, variance);
    latent = session.transpose(latent, {1, 2, 0});
    latent = session.add(session.multiply(latent, std_tensor), mean_tensor);
  }
  const uint32_t latent_height = latent.shape[0];
  const uint32_t latent_width = latent.shape[1];
  const auto ys = tile_positions(latent_height);
  const auto xs = tile_positions(latent_width);
  const uint32_t width = latent_width * 16;
  const uint32_t height = latent_height * 16;
  std::vector<float> canvas(width * height * 3, 0.0f);
  Tensor ones = float_tensor({64}, std::vector<float>(64, 1.0f));
  auto [cosine, sine] = vae_cosine_sine();
  // Tiles decode in batches, in raster order. A batch shares the decoder weights and
  // multiplies the rows of every matrix product, where one tile at a time left the GPU
  // bounded by the dispatch of each small operation.
  const float* latent_values = static_cast<const float*>(latent.buffer.contents);
  const size_t tile_count = ys.size() * xs.size();
  const size_t batch = decode_batch();
  std::vector<float> strip;  // The row of tiles being placed, before it joins the canvas.
  size_t strip_row = 0;
  const auto close_row = [&] {
    std::copy(strip.begin(), strip.end(), canvas.begin() + 16 * ys[strip_row] * width * 3);
  };
  for (size_t first = 0; first < tile_count; first += batch) {
    @autoreleasepool {
      const size_t count = std::min(batch, tile_count - first);
      std::vector<int32_t> positions;
      std::vector<float> patches(count * 16 * 16 * 24);
      for (size_t index = 0; index < count; ++index) {
        const int32_t y = ys[(first + index) / xs.size()];
        const int32_t x = xs[(first + index) % xs.size()];
        positions.push_back(y);
        positions.push_back(x);
        float* destination = patches.data() + index * 16 * 16 * 24;
        for (uint32_t dy = 0; dy < 16; ++dy) {
          std::copy_n(latent_values + ((y + dy) * latent_width + x) * 24, 16 * 24, destination + dy * 16 * 24);
        }
      }
      Tensor decoded = decode_tile(session, store, float_tensor({static_cast<int64_t>(count), 16, 16, 24}, patches), ones, cosine, sine);
      const float* values = static_cast<const float*>(decoded.buffer.contents);
      for (size_t index = 0; index < count; ++index) {
        const size_t tile_index = first + index;
        const size_t row = tile_index / xs.size();
        const size_t column = tile_index % xs.size();
        // Raster order finishes one row of tiles when the next one starts.
        if (column == 0) {
          if (tile_index > 0) close_row();
          strip.assign(256 * width * 3, 0.0f);
          strip_row = row;
        }
        const int32_t y = positions[2 * index];
        const int32_t x = positions[2 * index + 1];
        const float* tile = values + index * 256 * 256 * 3;
        const int32_t vertical = row ? (ys[row - 1] + 16 - y) * 16 : 0;
        const int32_t horizontal = column ? (xs[column - 1] + 16 - x) * 16 : 0;
        for (uint32_t py = 0; py < 256; ++py) {
          for (uint32_t px = 0; px < 256; ++px) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
              float value = tile[(py * 256 + px) * 3 + channel];
              const size_t target = (py * width + 16 * x + px) * 3 + channel;
              if (py < static_cast<uint32_t>(vertical)) {
                const size_t old_index = ((16 * y + py) * width + 16 * x + px) * 3 + channel;
                const float alpha = static_cast<float>(py) / vertical;
                value = canvas[old_index] + (value - canvas[old_index]) * alpha;
              }
              if (px < static_cast<uint32_t>(horizontal)) {
                const float alpha = static_cast<float>(px) / horizontal;
                value = strip[target] + (value - strip[target]) * alpha;
              }
              strip[target] = value;
            }
          }
        }
      }
    }
  }
  close_row();
  const float mean[] = {0.485f, 0.456f, 0.406f};
  const float variance[] = {0.229f, 0.224f, 0.225f};
  std::vector<uint8_t> rgb(canvas.size());
  for (size_t index = 0; index < canvas.size(); ++index) {
    const size_t channel = index % 3;
    const float value = canvas[index] * variance[channel] + mean[channel];
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite VAE output");
    rgb[index] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
  }
  write_png(output, width, height, rgb);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"vae\",\"profile\":\"%s\",\"weights\":\"%s\",\"seconds\":%.6f,\"output\":\"%s\","
              "\"tiles\":%zu,\"decode_batch\":%zu}\n",
              options.profile.c_str(), options.weights.c_str(), seconds, output.c_str(), tile_count,
              std::min(batch, tile_count));
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
  }
  return latent;
}

}  // namespace onyx
