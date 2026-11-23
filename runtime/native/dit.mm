#import "native.hpp"
#import <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <vector>

namespace onyx {
namespace {

constexpr uint32_t kDiTBlocks = 50;
// Weights one DiT block consumes, in the order the block reads them.
const std::vector<std::string> kBlockTensors = {
    "norm1.weight", "attn.qkv_proj.weight", "attn.q_norm.weight", "attn.k_norm.weight",
    "attn.out_proj.weight", "norm2.weight", "mlp.fc1.weight", "mlp.fc2.weight"};
// Weight and bias appended to a linear-layer name, such as an AdaLN projection.
const std::vector<std::string> kLinearSuffixes = {".weight", ".bias"};
const std::vector<int64_t> kBlockModulationShape = {6, 3, 6, 5376};
const std::vector<int64_t> kFinalModulationShape = {6, 1, 2, 5376};
constexpr size_t kBlockModulationBytes = 6 * 3 * 6 * 5376 * sizeof(uint16_t);
constexpr size_t kFinalModulationBytes = 6 * 2 * 5376 * sizeof(uint16_t);
constexpr size_t kModulationBytes = kDiTBlocks * kBlockModulationBytes + kFinalModulationBytes;

Tensor make_tensor(const std::vector<int64_t>& shape, MPSDataType dtype, const std::vector<float>& values) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (dtype == MPSDataTypeBFloat16) {
    std::vector<uint16_t> data;
    data.reserve(values.size());
    for (float value : values) data.push_back(float_to_bf16(value));
    id<MTLBuffer> buffer = [device newBufferWithBytes:data.data() length:data.size() * 2 options:MTLResourceStorageModeShared];
    return Tensor{buffer, shape, dtype};
  }
  id<MTLBuffer> buffer = [device newBufferWithBytes:values.data() length:values.size() * sizeof(float) options:MTLResourceStorageModeShared];
  return Tensor{buffer, shape, dtype};
}

Tensor scalar(float value, MPSDataType dtype) {
  return make_tensor({1}, dtype, {value});
}

std::vector<float> read_sigmas(const std::string& model) {
  const auto path = std::filesystem::path(model).parent_path() / "pdd_config.json";
  @autoreleasepool {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (!data) throw std::runtime_error("Missing PDD schedule: " + path.string());
    NSError* error = nil;
    NSDictionary* root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!root || error) throw std::runtime_error("Invalid PDD schedule");
    std::vector<float> result;
    for (NSNumber* value in root[@"sigmas"]) result.push_back(value.floatValue);
    if (result.size() != 7) throw std::runtime_error("Expected six PDD steps");
    return result;
  }
}

Tensor timestep_projection(const std::vector<float>& sigmas) {
  std::vector<float> values(6 * 256);
  for (uint32_t step = 0; step < 6; ++step) {
    const float timestep = 1.0f - sigmas[step];
    for (uint32_t index = 0; index < 128; ++index) {
      const double angle = timestep * std::exp(-std::log(10000.0) * index / 128.0);
      values[step * 256 + index] = std::cos(angle);
      values[step * 256 + 128 + index] = std::sin(angle);
    }
  }
  return make_tensor({6, 256}, MPSDataTypeFloat32, values);
}

Tensor original_projection(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& temb, bool final) {
  Tensor activated = session.silu(temb);
  activated = session.cast(activated, MPSDataTypeBFloat16);
  Tensor value = session.linear(store, prefix + ".adaln_proj.linear", activated);
  return session.reshape(value, final ? kFinalModulationShape : kBlockModulationShape);
}

Tensor time_embedding(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas, bool prefetch) {
  Tensor temb = timestep_projection(sigmas);
  if (prefetch) store.prefetch("time_embedder.", {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias"});
  temb = session.linear(store, "time_embedder.proj_in", temb);
  temb = session.silu(temb);
  return session.linear(store, "time_embedder.proj_out", temb);
}

// The AdaLN table of block `layer`, or of the final layer when layer == kDiTBlocks.
Tensor modulation_table(GraphSession& session, WeightStore& store, uint32_t layer, const Tensor& temb) {
  return layer < kDiTBlocks
      ? original_projection(session, store, "blocks." + std::to_string(layer), temb, false)
      : original_projection(session, store, "final_layer", temb, true);
}

// Queue the weights of table `layer`. Nothing is queued past the final layer.
void queue_modulation_table(WeightStore& store, uint32_t layer) {
  if (layer > kDiTBlocks) return;
  store.prefetch(layer < kDiTBlocks ? "blocks." + std::to_string(layer) + ".adaln_proj.linear"
                                    : "final_layer.adaln_proj.linear",
                 kLinearSuffixes);
}

std::vector<Tensor> compute_modulations(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas,
                                        bool prefetch) {
  const Tensor temb = time_embedding(session, store, sigmas, prefetch);
  // Each projection streams a 520 MB BF16 weight. Two stay in flight so that the reads
  // overlap instead of pacing the projections.
  if (prefetch) {
    queue_modulation_table(store, 0);
    queue_modulation_table(store, 1);
  }
  std::vector<Tensor> modulations;
  for (uint32_t layer = 0; layer <= kDiTBlocks; ++layer) {
    @autoreleasepool {
      if (prefetch) queue_modulation_table(store, layer + 2);
      modulations.push_back(modulation_table(session, store, layer, temb));
    }
  }
  return modulations;
}

using Digest = std::array<uint8_t, CC_SHA256_DIGEST_LENGTH>;

Digest sha256(const void* bytes, size_t length) {
  Digest digest;
  CC_SHA256(bytes, static_cast<CC_LONG>(length), digest.data());
  return digest;
}

bool modulation_weights_are_raw(const WeightStore& store) {
  for (const auto& name : store.names()) {
    if ((name.starts_with("time_embedder.") || name.find(".adaln_proj.") != std::string::npos) &&
        store.dtype(name).starts_with("Q4")) {
      return false;
    }
  }
  return true;
}

Digest modulation_cache_key(id<MTLDevice> device, const std::string& weights, const std::vector<float>& sigmas,
                            NSData* binary) {
  const auto build = sha256(binary.bytes, binary.length);
  NSData* device_and_os = [[NSString stringWithFormat:@"%@\n%@",
      device.name, NSProcessInfo.processInfo.operatingSystemVersionString] dataUsingEncoding:NSUTF8StringEncoding];
  NSMutableData* key = [NSMutableData dataWithBytes:build.data() length:build.size()];
  [key appendBytes:sigmas.data() length:sigmas.size() * sizeof(float)];
  [key appendBytes:weights.c_str() length:weights.size() + 1];
  [key appendData:device_and_os];
  return sha256(key.bytes, key.length);
}

NSString* modulation_cache_path(NSString* directory, const Digest& key) {
  constexpr char hex[] = "0123456789abcdef";
  std::string name;
  for (uint8_t byte : key) {
    name += hex[byte >> 4];
    name += hex[byte & 15];
  }
  name += ".adaln";
  return [directory stringByAppendingPathComponent:[NSString stringWithUTF8String:name.c_str()]];
}

std::vector<Tensor> load_modulations(id<MTLDevice> device, NSString* path, const Digest& key) {
  NSData* data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil];
  if (data.length != 2 * key.size() + kModulationBytes) return {};
  const auto* bytes = static_cast<const uint8_t*>(data.bytes);
  if (std::memcmp(bytes, key.data(), key.size()) != 0) return {};
  const auto* payload = bytes + 2 * key.size();
  const auto checksum = sha256(payload, kModulationBytes);
  if (std::memcmp(bytes + key.size(), checksum.data(), checksum.size()) != 0) return {};

  id<MTLBuffer> buffer = [device newBufferWithBytes:payload length:kModulationBytes options:MTLResourceStorageModeShared];
  if (!buffer) return {};
  std::vector<Tensor> result;
  for (uint32_t layer = 0; layer < kDiTBlocks; ++layer) {
    result.push_back({buffer, kBlockModulationShape, MPSDataTypeBFloat16, layer * kBlockModulationBytes});
  }
  result.push_back({buffer, kFinalModulationShape, MPSDataTypeBFloat16, kDiTBlocks * kBlockModulationBytes});
  return result;
}

bool save_modulations(NSString* directory, NSString* path, const Digest& key, const std::vector<Tensor>& modulations) {
  if (![NSFileManager.defaultManager createDirectoryAtPath:directory withIntermediateDirectories:YES
                                               attributes:@{NSFilePosixPermissions: @0700} error:nil]) return false;
  // The cache format stores the key and payload digest before the packed BF16 tables.
  NSMutableData* data = [NSMutableData dataWithLength:2 * key.size() + kModulationBytes];
  auto* bytes = static_cast<uint8_t*>(data.mutableBytes);
  std::memcpy(bytes, key.data(), key.size());
  size_t offset = 2 * key.size();
  for (const Tensor& tensor : modulations) {
    std::memcpy(bytes + offset, tensor.data(), tensor.byte_size());
    offset += tensor.byte_size();
  }
  const auto checksum = sha256(bytes + 2 * key.size(), kModulationBytes);
  std::memcpy(bytes + key.size(), checksum.data(), checksum.size());
  return [data writeToFile:path options:NSDataWritingAtomic error:nil];
}

std::vector<Tensor> get_modulations(GraphSession& session, WeightStore& store, const std::vector<float>& sigmas,
                                   id<MTLDevice> device, const std::string& cache_dir, bool prefetch, const char*& cache_status) {
  if (cache_dir.empty()) {
    cache_status = "disabled";
    return compute_modulations(session, store, sigmas, prefetch);
  }
  @autoreleasepool {
    NSString* directory = [NSString stringWithUTF8String:cache_dir.c_str()];
    NSString* binary_path = NSBundle.mainBundle.executablePath;
    NSData* binary = binary_path ? [NSData dataWithContentsOfFile:binary_path options:NSDataReadingMappedIfSafe error:nil] : nil;
    const std::string weights = store.file_identity();
    if (!directory || !binary.length || binary.length > UINT32_MAX || weights.empty() || !modulation_weights_are_raw(store)) {
    cache_status = "unavailable";
    return compute_modulations(session, store, sigmas, prefetch);
  }

    const auto key = modulation_cache_key(device, weights, sigmas, binary);
    NSString* path = modulation_cache_path(directory, key);
    auto result = load_modulations(device, path, key);
    if (!result.empty()) {
      cache_status = "hit";
      return result;
    }

    result = compute_modulations(session, store, sigmas, prefetch);
    cache_status = weights == store.file_identity() && save_modulations(directory, path, key, result) ? "miss" : "write_failed";
    return result;
  }
}

Tensor qkv_attention(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& x, const Tensor* cosine, const Tensor* sine) {
  Tensor fused = session.linear(store, prefix + ".qkv_proj", x);
  Tensor result = session.fused_qkv_attention(fused, store.tensor(prefix + ".q_norm.weight"),
                                               store.tensor(prefix + ".k_norm.weight"), cosine, sine);
  return session.linear(store, prefix + ".out_proj", result);
}

Tensor mlp(GraphSession& session, WeightStore& store, const std::string& prefix, const Tensor& x) {
  Tensor hidden = session.linear(store, prefix + ".fc1", x);
  return session.linear(store, prefix + ".fc2", session.gated_activation(hidden));
}

Tensor prepare_text(GraphSession& session, WeightStore& store, const Tensor& embedding) {
  Tensor x = session.linear(store, "condition_proj", embedding);
  for (uint32_t layer = 0; layer < 2; ++layer) {
    @autoreleasepool {
      const std::string prefix = "token_refiner.blocks." + std::to_string(layer) + ".";
      Tensor h = session.rms_norm(x, store.tensor(prefix + "norm1.weight"), 1e-5);
      x = session.add(x, qkv_attention(session, store, prefix + "attn", h, nullptr, nullptr));
      h = session.rms_norm(x, store.tensor(prefix + "norm2.weight"), 1e-5);
      x = session.add(x, mlp(session, store, prefix + "mlp", h));
    }
  }
  return session.rms_norm(x, store.tensor("token_refiner.final_norm.weight"), 1e-5);
}

std::pair<Tensor, Tensor> rope_angles(WeightStore& store, uint64_t text_count, uint32_t rows, uint32_t columns) {
  const uint64_t count = text_count + rows * columns;
  Tensor frequencies = store.tensor("rope.inv_freq");
  const float* frequency = static_cast<const float*>(frequencies.buffer.contents);
  std::vector<float> cosine(count * 96);
  std::vector<float> sine(count * 96);
  const double area = std::sqrt(static_cast<double>(rows * columns));
  for (uint64_t row = 0; row < count; ++row) {
    double position[3] = {0, 0, 0};
    if (row < text_count) {
      position[0] = row;
    } else {
      const uint64_t image = row - text_count;
      const uint32_t y = image / columns;
      const uint32_t x = image % columns;
      position[0] = text_count;
      position[1] = (y / area + (1 - static_cast<double>(rows) / area) / 2) * 32;
      position[2] = (x / area + (1 - static_cast<double>(columns) / area) / 2) * 32;
    }
    for (uint32_t axis = 0; axis < 3; ++axis) {
      for (uint32_t index = 0; index < 16; ++index) {
        const double angle = position[axis] * frequency[index];
        for (uint32_t copy = 0; copy < 2; ++copy) {
          const uint64_t output = row * 96 + copy * 48 + axis * 16 + index;
          cosine[output] = std::cos(angle);
          sine[output] = std::sin(angle);
        }
      }
    }
  }
  const std::vector<int64_t> shape = {1, 1, static_cast<int64_t>(count), 96};
  return {make_tensor(shape, MPSDataTypeBFloat16, cosine), make_tensor(shape, MPSDataTypeBFloat16, sine)};
}

std::vector<Tensor> modulation_parts(const Tensor& modulation, const Tensor& indices) {
  const uint64_t rows = indices.shape[0];
  const uint64_t row_bytes = 5376 * sizeof(uint16_t);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  const uint16_t* source = static_cast<const uint16_t*>(modulation.data());
  const int32_t* modalities = static_cast<const int32_t*>(indices.buffer.contents);
  std::vector<Tensor> result(6, Tensor{nil, {static_cast<int64_t>(rows), 5376}, MPSDataTypeBFloat16});
  for (uint64_t part = 0; part < 6; ++part) {
    Tensor& output = result[part];
    output.buffer = [device newBufferWithLength:output.byte_size() options:MTLResourceStorageModeShared];
    if (!output.buffer) throw std::runtime_error("Modulation allocation failed");
    uint16_t* destination = static_cast<uint16_t*>(output.buffer.contents);
    for (uint64_t row = 0; row < rows; ++row) {
      const uint64_t source_index = (static_cast<uint64_t>(modalities[row]) * 6 + part) * 5376;
      std::memcpy(destination + row * 5376, source + source_index, row_bytes);
    }
  }
  return result;
}

}  // namespace

Tensor denoise_image(const std::string& model, const std::string& embedding_path, const std::string& noise_path, const std::string& pdd_path, const std::string& output, uint32_t width, uint32_t height, const StageOptions& options) {
  if (width % 32 || height % 32) throw std::runtime_error("H3 dimensions must be multiples of 32");
  require_streaming_weights(options, "dit", "W4 DiT pack 37.47 GB exceeds physical memory");
  if (options.prefetch_weights && options.cache_bytes != 0) {
    throw std::runtime_error("DiT prefetch requires --cache-bytes 0");
  }
  const auto began = std::chrono::steady_clock::now();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  GraphSession session(device);
  if (std::getenv("ONYX_PROFILE")) session.enable_profiling(true);
  WeightStore store(model, false, options.cache_bytes);
  WeightStore pdd(pdd_path);
  const auto sigmas = read_sigmas(model);
  const char* modulation_cache_status = nullptr;
  const auto modulation_began = std::chrono::steady_clock::now();
  // Without an on-disk modulation cache, the first sampling step computes each table right
  // before its block uses it. The 520 MB projection weights then stream alongside the block
  // weights instead of in a serial pass ahead of sampling.
  const bool lazy_modulation = options.modulation_cache_dir.empty() && options.prefetch_weights;
  std::vector<Tensor> modulations;
  Tensor temb{};
  if (lazy_modulation) {
    modulation_cache_status = "disabled";
    temb = time_embedding(session, store, sigmas, true);
    modulations.assign(kDiTBlocks + 1, Tensor{});
    queue_modulation_table(store, 0);
    queue_modulation_table(store, 1);
  } else {
    modulations = get_modulations(session, store, sigmas, device, options.modulation_cache_dir,
                                  options.prefetch_weights, modulation_cache_status);
  }
  const double modulation_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - modulation_began).count();
  const Tensor& final_modulation = modulations.back();
  Tensor text = prepare_text(session, store, load_tensor(embedding_path, "embedding"));
  const uint32_t rows = height / 32;
  const uint32_t columns = width / 32;
  const uint64_t text_count = text.shape[0];
  const uint64_t image_count = rows * columns;
  const uint64_t sequence_count = text_count + image_count;
  auto [cosine, sine] = rope_angles(store, text_count, rows, columns);
  Tensor indices = make_tensor({static_cast<int64_t>(sequence_count), 1}, MPSDataTypeInt32, std::vector<float>(sequence_count, 0));
  auto* index_data = static_cast<int32_t*>(indices.buffer.contents);
  for (uint64_t index = 0; index < sequence_count; ++index) index_data[index] = index < text_count ? 1 : 0;
  const Tensor one = scalar(1.0f, MPSDataTypeBFloat16);
  Tensor patches = load_tensor(noise_path, "latent");
  patches = session.reshape(patches, {24, static_cast<int64_t>(rows), 2, static_cast<int64_t>(columns), 2});
  patches = session.transpose(patches, {1, 3, 0, 2, 4});
  patches = session.reshape(patches, {static_cast<int64_t>(image_count), 96});
  // Keep the reader one block ahead. The pipeline stays full across sampling-step boundaries,
  // where the next step's patch projection and block 0 follow the last block.
  const auto queue_block = [&](uint32_t layer) {
    if (!options.prefetch_weights) return;
    if (layer == 0) store.prefetch("video_patch_proj", kLinearSuffixes);
    store.prefetch("blocks." + std::to_string(layer) + ".", kBlockTensors);
  };
  queue_block(0);
  for (uint32_t step = 0; step < 6; ++step) {
    @autoreleasepool {
    const auto step_began = std::chrono::steady_clock::now();
    Tensor image = session.linear(store, "video_patch_proj", patches);
    image = session.cast(image, MPSDataTypeBFloat16);
    Tensor x = session.concat({text, image}, 0);
    for (uint32_t layer = 0; layer < kDiTBlocks; ++layer) {
      @autoreleasepool {
      if (layer + 1 < kDiTBlocks || step + 1 < 6) queue_block((layer + 1) % kDiTBlocks);
      if (lazy_modulation && step == 0) {
        // The first step fills the tables; the other five reuse them.
        queue_modulation_table(store, layer + 2);
        if (!modulations[layer].buffer) modulations[layer] = modulation_table(session, store, layer, temb);
      }
      const std::string prefix = "blocks." + std::to_string(layer) + ".";
      Tensor modulation = modulations[layer];
      modulation.byte_offset += modulation.byte_size() / 6 * step;
      modulation.shape = {3, 6, 5376};
      const auto parts = modulation_parts(modulation, indices);
      const Tensor& shift_msa = parts[0];
      const Tensor& scale_msa = parts[1];
      const Tensor& gate_msa = parts[2];
      const Tensor& shift_mlp = parts[3];
      const Tensor& scale_mlp = parts[4];
      const Tensor& gate_mlp = parts[5];
      Tensor h = session.modulated_rms_norm(x, store.tensor(prefix + "norm1.weight"), scale_msa, shift_msa, 1e-5);
      Tensor attended = qkv_attention(session, store, prefix + "attn", h, &cosine, &sine);
      x = session.gated_residual(x, attended, gate_msa);
      h = session.modulated_rms_norm(x, store.tensor(prefix + "norm2.weight"), scale_mlp, shift_mlp, 1e-5);
      Tensor hidden = mlp(session, store, prefix + "mlp", h);
      x = session.gated_residual(x, hidden, gate_mlp);
      }
    }
    if (lazy_modulation && step == 0 && !modulations[kDiTBlocks].buffer) {
      modulations[kDiTBlocks] = modulation_table(session, store, kDiTBlocks, temb);
    }
    Tensor target = session.slice(x, 0, text_count, image_count);
    target = session.rms_norm(target, store.tensor("final_layer.norm.weight"), 1e-5);
    Tensor final = session.reshape(session.slice(final_modulation, 0, step, 1), {2, 5376});
    Tensor shift = session.slice(final, 0, 0, 1);
    Tensor scale = session.slice(final, 0, 1, 1);
    target = session.add(session.multiply(target, session.add(one, scale)), shift);
    target = session.cast(target, MPSDataTypeFloat32);
    Tensor head = session.reshape(session.slice(pdd.tensor("video_out.weight"), 0, step, 1), {96, 5376});
    Tensor bias = session.reshape(session.slice(pdd.tensor("video_out.bias"), 0, step, 1), {96});
    Tensor velocity = session.matmul(target, head);
    velocity = session.add(velocity, bias);
    Tensor delta = scalar(sigmas[step] - sigmas[step + 1], MPSDataTypeFloat32);
    patches = session.add(patches, session.multiply(velocity, delta));
    const double step_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - step_began).count();
    std::fprintf(stderr, "native DiT step %u/6, %.3f s, Metal allocated %.2f GiB\n", step + 1, step_seconds,
                 device.currentAllocatedSize / 1073741824.0);
    }
  }
  if (options.prefetch_weights) store.finish_prefetch();
  Tensor latent = session.reshape(patches, {static_cast<int64_t>(rows), static_cast<int64_t>(columns), 24, 2, 2});
  latent = session.transpose(latent, {2, 0, 3, 1, 4});
  latent = session.reshape(latent, {24, static_cast<int64_t>(rows * 2), static_cast<int64_t>(columns * 2)});
  save_tensor(output, "latent", latent);
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
  std::printf("{\"stage\":\"dit\",\"profile\":\"%s\",\"steps\":6,\"seconds\":%.6f,\"output\":\"%s\","
              "\"modulation_cache\":\"%s\",\"modulation_seconds\":%.6f,\"lazy_modulation\":%s,"
              "\"prefetch_weights\":%s,\"prefetch_wait_seconds\":%.6f,"
              "\"weight_cache\":{\"budget\":%llu,\"bytes\":%llu,\"entries\":%llu,\"hits\":%llu,\"read_calls\":%llu,\"read_bytes\":%llu}}\n",
              options.profile.c_str(), seconds, output.c_str(), modulation_cache_status, modulation_seconds,
              lazy_modulation ? "true" : "false",
              options.prefetch_weights ? "true" : "false", store.prefetch_wait_seconds(),
              static_cast<unsigned long long>(store.cache_budget()),
              static_cast<unsigned long long>(store.cache_bytes()),
              static_cast<unsigned long long>(store.cache_entries()),
              static_cast<unsigned long long>(store.cache_hits()),
              static_cast<unsigned long long>(store.read_calls()),
              static_cast<unsigned long long>(store.bytes_read()));
  if (std::getenv("ONYX_PROFILE")) {
    std::printf("{\"profile\":%s,\"weight_read\":{\"bytes\":%llu,\"calls\":%llu,\"seconds\":%.6f}}\n",
                session.profile_json().c_str(), store.bytes_read(), store.read_calls(), store.read_seconds());
    std::vector<std::pair<std::string, uint64_t>> reads(store.read_bytes_by_name().begin(), store.read_bytes_by_name().end());
    std::sort(reads.begin(), reads.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t index = 0; index < std::min<size_t>(reads.size(), 8); ++index) {
      std::fprintf(stderr, "read-bytes %s %llu\n", reads[index].first.c_str(), reads[index].second);
    }
  }
  return latent;
}

}  // namespace onyx
