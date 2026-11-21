#import "native.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace onyx {
namespace {

NSDictionary* json_object(NSData* data) {
  NSError* error = nil;
  id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  if (!value || ![value isKindOfClass:[NSDictionary class]]) {
    throw std::runtime_error("Invalid model metadata JSON");
  }
  return value;
}

NSData* read_at(int fd, uint64_t offset, uint64_t length) {
  NSMutableData* data = [NSMutableData dataWithLength:length];
  uint64_t completed = 0;
  while (completed < length) {
    const ssize_t result = pread(fd, static_cast<uint8_t*>(data.mutableBytes) + completed, length - completed, offset + completed);
    if (result < 0) throw std::runtime_error(std::strerror(errno));
    if (result == 0) throw std::runtime_error("Unexpected end of model file");
    completed += result;
  }
  return data;
}

MTLResourceOptions resource_options = MTLResourceStorageModeShared;

}  // namespace

uint64_t Tensor::elements() const {
  uint64_t result = 1;
  for (int64_t dimension : shape) result *= dimension;
  return result;
}

uint64_t Tensor::byte_size() const {
  const uint64_t count = elements();
  if (dtype == MPSDataTypeUInt4 || dtype == MPSDataTypeInt4) return count / 2;
  if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeBool) return count;
  if (dtype == MPSDataTypeFloat16 || dtype == MPSDataTypeBFloat16 || dtype == MPSDataTypeInt16 || dtype == MPSDataTypeUInt16) return count * 2;
  return count * 4;
}

void* Tensor::data() const {
  return static_cast<uint8_t*>(buffer.contents) + byte_offset;
}

MPSDataType data_type(const std::string& name) {
  if (name == "BF16") return MPSDataTypeBFloat16;
  if (name == "F16") return MPSDataTypeFloat16;
  if (name == "F32") return MPSDataTypeFloat32;
  if (name == "I8") return MPSDataTypeInt8;
  if (name == "U8") return MPSDataTypeUInt8;
  if (name == "I64") return MPSDataTypeInt64;
  if (name == "I32") return MPSDataTypeInt32;
  if (name == "U32") return MPSDataTypeUInt32;
  if (name == "BOOL") return MPSDataTypeBool;
  if (name.starts_with("Q4S")) return MPSDataTypeUInt4;
  throw std::runtime_error("Unsupported dtype: " + name);
}

std::vector<int64_t> shape_of(NSArray* values) {
  std::vector<int64_t> result;
  for (NSNumber* value in values) result.push_back(value.longLongValue);
  return result;
}

StageOptions parse_stage_options(int argc, char** argv) {
  StageOptions options;
  bool have_profile = false;
  bool have_weights = false;
  for (int index = 2; index + 1 < argc; ++index) {
    const std::string name = argv[index];
    if (name == "--profile") {
      options.profile = argv[index + 1];
      have_profile = true;
    }
    if (name == "--weights") {
      options.weights = argv[index + 1];
      have_weights = true;
    }
  }
  if (!have_profile) throw std::runtime_error("Missing required argument: --profile");
  if (!have_weights) throw std::runtime_error("Missing required argument: --weights");
  static const std::string allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-";
  if (options.profile.empty() || options.profile.size() > 64) throw std::runtime_error("Invalid profile id: " + options.profile);
  for (const char character : options.profile) {
    if (allowed.find(character) == std::string::npos) throw std::runtime_error("Invalid profile id: " + options.profile);
  }
  if (options.weights != "resident" && options.weights != "stream") {
    throw std::runtime_error("Invalid weights mode: " + options.weights + " (expected resident|stream)");
  }
  return options;
}

void require_streaming_weights(const StageOptions& options, const char* stage, const char* reason) {
  if (options.weights != "stream") {
    throw std::runtime_error(std::string(stage) + " stage requires --weights stream (" + reason + "); requested " + options.weights);
  }
}

WeightStore::WeightStore(const std::string& path, bool resident, uint64_t cache_budget_bytes)
    : fd_(-1), data_begin_(0), specs_(nil), resident_(resident), cache_budget_(resident ? UINT64_MAX : cache_budget_bytes), cache_() {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("Could not open weights: " + path);
  @autoreleasepool {
    NSData* prefix = read_at(fd_, 0, 8);
    uint64_t length = 0;
    std::memcpy(&length, prefix.bytes, 8);
    const bool packed = std::memcmp(prefix.bytes, "ONYXPK01", 8) == 0;
    if (packed) {
      NSData* size_data = read_at(fd_, 8, 8);
      std::memcpy(&length, size_data.bytes, 8);
      data_begin_ = 16 + length;
      NSData* header = read_at(fd_, 16, length);
      NSDictionary* root = json_object(header);
      NSMutableDictionary* specs = [NSMutableDictionary new];
      for (NSDictionary* spec in root[@"tensors"]) specs[spec[@"name"]] = spec;
      specs_ = specs;
    } else {
      data_begin_ = 8 + length;
      specs_ = json_object(read_at(fd_, 8, length));
    }
  }
}

WeightStore::~WeightStore() {
  if (prefetch_.valid()) prefetch_.wait();
  if (fd_ >= 0) close(fd_);
}

std::string WeightStore::file_identity() const {
  struct stat info {};
  if (fstat(fd_, &info) != 0) return {};
  return std::to_string(info.st_dev) + ":" + std::to_string(info.st_ino) + ":" + std::to_string(info.st_size) +
         ":" + std::to_string(info.st_mtimespec.tv_sec) + ":" + std::to_string(info.st_mtimespec.tv_nsec) +
         ":" + std::to_string(info.st_ctimespec.tv_sec) + ":" + std::to_string(info.st_ctimespec.tv_nsec);
}

bool WeightStore::has(const std::string& name) const {
  return specs_[[NSString stringWithUTF8String:name.c_str()]] != nil;
}

std::string WeightStore::dtype(const std::string& name) const {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown tensor: " + name);
  return [spec[@"dtype"] UTF8String];
}

std::vector<std::string> WeightStore::names() const {
  std::vector<std::string> result;
  result.reserve([specs_ count]);
  for (NSString* key in specs_) result.push_back([key UTF8String]);
  std::sort(result.begin(), result.end());
  return result;
}

void WeightStore::touch(const std::string& name) {
  auto position = lru_order_.begin();
  // Move an existing block to the MRU end. The list is bounded by the number of cached blocks.
  for (position = lru_order_.begin(); position != lru_order_.end(); ++position) {
    if (*position == name) break;
  }
  if (position != lru_order_.end()) lru_order_.splice(lru_order_.begin(), lru_order_, position);
}

void WeightStore::admit(const std::string& name, const Tensor& value) {
  const uint64_t bytes = value.buffer.length;
  if (bytes > cache_budget_ || cache_.count(name)) return;
  while (cache_bytes_ + bytes > cache_budget_ && !lru_order_.empty()) {
    const std::string victim = lru_order_.back();
    const uint64_t victim_bytes = lru_bytes_[victim];
    cache_bytes_ -= victim_bytes;
    lru_bytes_.erase(victim);
    lru_order_.pop_back();
    cache_.erase(victim);
  }
  cache_[name] = value;
  lru_bytes_[name] = bytes;
  cache_bytes_ += bytes;
  lru_order_.push_front(name);
}

Tensor WeightStore::tensor(const std::string& name) {
  auto prefetched = prefetched_.extract(name);
  if (!prefetched.empty()) return std::move(prefetched.mapped());
  auto found = cache_.find(name);
  if (found != cache_.end()) {
    if (!resident_) {
      ++cache_hits_;
      touch(name);
    }
    return found->second;
  }
  Tensor result = read(name, "data_offsets", false);
  if (resident_) {
    cache_[name] = result;
  } else if (cache_budget_ > 0) {
    admit(name, result);
  }
  return result;
}

void WeightStore::prefetch(std::vector<std::string> names) {
  if (cache_budget_ != 0 || prefetch_.valid()) {
    throw std::logic_error("Weight prefetch requires streaming and at most one pending batch");
  }
  prefetch_ = std::async(std::launch::async, [this, names = std::move(names)] {
    std::map<std::string, Tensor> result;
    for (const auto& name : names) {
      @autoreleasepool {
        result.emplace(name, read(name, "data_offsets", false));
      }
    }
    return result;
  });
}

void WeightStore::finish_prefetch() {
  const auto began = std::chrono::steady_clock::now();
  prefetched_ = prefetch_.get();
  prefetch_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - began).count();
}

std::vector<int64_t> WeightStore::shape(const std::string& name) const {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown tensor: " + name);
  return shape_of(spec[@"shape"]);
}

Tensor WeightStore::read(const std::string& name, const std::string& field, bool scales) {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown tensor: " + name);
  NSArray* offsets = spec[[NSString stringWithUTF8String:field.c_str()]];
  if (!offsets) throw std::runtime_error("Missing tensor offsets: " + name);
  const uint64_t begin = [offsets[0] unsignedLongLongValue];
  const uint64_t end = [offsets[1] unsignedLongLongValue];
  const uint64_t length = end - begin;
  const auto read_start = std::chrono::steady_clock::now();
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithLength:length options:resource_options];
  if (!buffer) throw std::runtime_error("Metal weight allocation failed: " + name);
  uint64_t completed = 0;
  while (completed < length) {
    const ssize_t result = pread(fd_, static_cast<uint8_t*>(buffer.contents) + completed, length - completed, data_begin_ + begin + completed);
    if (result < 0) throw std::runtime_error(std::strerror(errno));
    if (result == 0) throw std::runtime_error("Unexpected end of model file: " + name);
    completed += result;
  }
  {
    const std::lock_guard lock(read_stats_mutex_);
    bytes_read_ += length;
    read_bytes_by_name_[name] += length;
    ++read_calls_;
    read_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - read_start).count();
  }
  const std::string type = scales ? "F32" : dtype(name);
  return Tensor{buffer, shape(name), data_type(type)};
}

Tensor WeightStore::scale(const std::string& name) {
  const std::string key = name + "/scale";
  auto found = cache_.find(key);
  if (found != cache_.end()) return found->second;
  Tensor result = read(name, "scale_offsets", true);
  const std::string type = dtype(name);
  const uint64_t group = std::stoull(type.substr(3));
  result.shape = {result.shape[0], result.shape[1] / static_cast<int64_t>(group)};
  cache_[key] = result;
  return result;
}

Tensor WeightStore::embedding(const std::string& name, const std::vector<int32_t>& ids) {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown embedding: " + name);
  const std::string type = dtype(name);
  if (type.starts_with("Q4S")) throw std::runtime_error("Quantized embeddings are not supported by the native frontend");
  const auto shape = shape_of(spec[@"shape"]);
  NSArray* offsets = spec[@"data_offsets"];
  const uint64_t begin = [offsets[0] unsignedLongLongValue];
  const uint64_t end = [offsets[1] unsignedLongLongValue];
  const uint64_t stride = (end - begin) / shape[0];
  const uint64_t bytes = stride * ids.size();
  NSMutableData* data = [NSMutableData dataWithLength:bytes];
  for (size_t index = 0; index < ids.size(); ++index) {
    const uint64_t offset = data_begin_ + begin + stride * static_cast<uint32_t>(ids[index]);
    const ssize_t result = pread(fd_, static_cast<uint8_t*>(data.mutableBytes) + index * stride, stride, offset);
    if (result != static_cast<ssize_t>(stride)) throw std::runtime_error("Could not read embedding row");
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  id<MTLBuffer> buffer = [device newBufferWithBytes:data.bytes length:data.length options:resource_options];
  return Tensor{buffer, {static_cast<int64_t>(ids.size()), shape[1]}, data_type(type)};
}

uint16_t float_to_fp16_bits(float value) {
  __fp16 bits = value;
  return *reinterpret_cast<uint16_t*>(&bits);
}

float fp16_bits_to_float(uint16_t bits) {
  __fp16 value = *reinterpret_cast<__fp16*>(&bits);
  return float(value);
}

uint16_t float_to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float bf16_to_float(uint16_t value) {
  uint32_t bits = uint32_t(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void save_tensor(const std::string& path, const std::string& name, const Tensor& value) {
  @autoreleasepool {
    const uint64_t bytes = value.byte_size();
    NSMutableArray* shape = [NSMutableArray new];
    for (int64_t dimension : value.shape) [shape addObject:@(dimension)];
    NSDictionary* root = @{[NSString stringWithUTF8String:name.c_str()]: @{ @"dtype": value.dtype == MPSDataTypeBFloat16 ? @"BF16" : value.dtype == MPSDataTypeFloat16 ? @"F16" : @"F32", @"shape": shape, @"data_offsets": @[@0, @(bytes)] }};
    NSData* header = [NSJSONSerialization dataWithJSONObject:root options:0 error:nil];
    int fd = open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) throw std::runtime_error("Could not create tensor artifact: " + path);
    const uint64_t length = header.length;
    if (write(fd, &length, 8) != 8 || write(fd, header.bytes, header.length) != static_cast<ssize_t>(header.length)) {
      close(fd);
      throw std::runtime_error("Could not write tensor header");
    }
    const uint64_t written = write(fd, value.data(), bytes);
    close(fd);
    if (written != bytes) throw std::runtime_error("Could not write tensor data");
  }
}

Tensor load_tensor(const std::string& path, const std::string& name) {
  WeightStore store(path);
  return store.tensor(name);
}

std::vector<int32_t> read_token_ids(const std::string& path) {
  @autoreleasepool {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (!data) throw std::runtime_error("Could not read token IDs: " + path);
    NSDictionary* root = json_object(data);
    std::vector<int32_t> result;
    for (NSNumber* value in root[@"ids"]) result.push_back(value.intValue);
    if (result.empty()) throw std::runtime_error("No token IDs");
    return result;
  }
}

}  // namespace onyx
