#import "native.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace onyx {
namespace {

// Reader threads per store. Cold model files deliver about 6.4 GiB/s on the M5 Max
// reference machine with any thread count, so a second reader only keeps small tensors
// from waiting behind a large one.
uint64_t reader_threads() {
  static const uint64_t count = [] {
    const char* value = std::getenv("ONYX_PREFETCH_THREADS");
    const long long parsed = value ? std::atoll(value) : 2;
    return parsed > 0 ? static_cast<uint64_t>(parsed) : 1;
  }();
  return count;
}

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
    : fd_(-1), data_begin_(0), specs_(nil), names_(), device_(MTLCreateSystemDefaultDevice()),
      resident_(resident), cache_budget_(resident ? UINT64_MAX : cache_budget_bytes), cache_() {
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
    for (NSString* key in specs_) names_.push_back(key.UTF8String);
    std::sort(names_.begin(), names_.end());
  }
}

WeightStore::~WeightStore() {
  {
    const std::lock_guard lock(prefetch_mutex_);
    prefetch_stop_ = true;
    prefetch_queue_.clear();
  }
  work_available_.notify_all();
  for (std::thread& reader : prefetch_readers_) reader.join();
  if (fd_ >= 0) close(fd_);
}

WeightStore::ForegroundRead& WeightStore::foreground_read(const std::string& name) {
  if (name.starts_with("blocks.")) {
    return foreground_reads_[name.substr(0, name.find('.', sizeof("blocks.") - 1))];
  }
  return foreground_reads_["other"];
}

std::string WeightStore::foreground_read_profile_json() const {
  NSMutableDictionary* blocks = [NSMutableDictionary new];
  for (const auto& [name, read] : foreground_reads_) {
    blocks[[NSString stringWithUTF8String:name.c_str()]] = @{
        @"direct_calls": @(read.direct_calls), @"direct_bytes": @(read.direct_bytes),
        @"direct_ns": @(read.direct_ns), @"blocked_prefetch_calls": @(read.blocked_calls),
        @"blocked_prefetch_ns": @(read.blocked_ns)};
  }
  NSDictionary* profile = @{@"groups": blocks, @"prefetch_drain_ns": @(prefetch_drain_ns_)};
  NSData* data = [NSJSONSerialization dataWithJSONObject:profile options:NSJSONWritingSortedKeys error:nil];
  if (!data) throw std::runtime_error("Could not encode foreground weight reads");
  return std::string(static_cast<const char*>(data.bytes), data.length);
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
  return names_;
}

void WeightStore::plan_shared_cache(const std::vector<std::vector<std::string>>& blocks,
                                    const std::vector<std::string>& step_boundary) {
  if (!cache_budget_ || resident_ || shared_cache_ || !cache_.empty() || !prefetch_readers_.empty()) {
    throw std::logic_error("Shared weight budget requires a fresh streaming store with a positive budget");
  }
  const auto allocations = [&](const std::vector<std::string>& names) {
    std::map<std::string, uint64_t> entries;
    uint64_t bytes = 0;
    for (const std::string& name : names) {
      if (!has(name)) continue;
      const uint64_t weight = read_range(name, "data_offsets").second;
      if (entries.emplace(name, weight).second) bytes += weight;
      if (dtype(name).starts_with("Q4")) {
        const uint64_t scale = read_range(name, "scale_offsets").second;
        if (entries.emplace(name + "/scale", scale).second) bytes += scale;
      }
    }
    return std::pair{std::move(entries), bytes};
  };
  std::vector<std::map<std::string, uint64_t>> block_allocations;
  uint64_t largest_block = 0;
  for (const auto& names : blocks) {
    auto [entries, bytes] = allocations(names);
    largest_block = std::max(largest_block, bytes);
    block_allocations.push_back(std::move(entries));
  }
  // A request can queue the next block before consuming all weights of the current one.
  const uint64_t prefetch_room = 2 * largest_block + allocations(step_boundary).second;
  const uint64_t resident_cap = cache_budget_ - std::min(cache_budget_, prefetch_room);
  const auto plan = [&](const std::string& key, uint64_t bytes) {
    if (planned_residents_.emplace(key, bytes).second) shared_resident_plan_bytes_ += bytes;
  };
  // Keep the small, repeatedly used scales ahead of full blocks.
  for (const auto& entries : block_allocations) {
    for (const auto& [key, bytes] : entries) {
      if (key.ends_with("/scale") && bytes <= resident_cap - shared_resident_plan_bytes_) plan(key, bytes);
    }
  }
  for (const auto& entries : block_allocations) {
    uint64_t additional = 0;
    for (const auto& [key, bytes] : entries) {
      if (!planned_residents_.contains(key)) additional += bytes;
    }
    if (entries.empty() || additional > resident_cap - shared_resident_plan_bytes_) continue;
    for (const auto& [key, bytes] : entries) plan(key, bytes);
    ++shared_resident_blocks_;
  }
  shared_cache_ = true;
}

void WeightStore::retain_shared(const std::string& key, const Tensor& value, bool scales) {
  const auto plan = planned_residents_.find(key);
  if (plan == planned_residents_.end()) return;
  const uint64_t bytes = value.buffer.length;
  if (bytes != plan->second || bytes > cache_budget_ - cache_bytes_ - pending_prefetch_bytes_) {
    throw std::logic_error("Shared weight store exceeded its resident plan");
  }
  cache_.emplace(key, value);
  cache_bytes_ += bytes;
  if (scales) scale_cache_bytes_ += bytes;
  peak_shared_bytes_ = std::max(peak_shared_bytes_, cache_bytes_ + pending_prefetch_bytes_);
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
  if (Tensor value = take_prefetched(name); value.buffer) {
    if (shared_cache_) retain_shared(name, value, false);
    return value;
  }
  if (std::getenv("ONYX_PREFETCH_TRACE")) {
    std::fprintf(stderr, "prefetch trace: synchronous read of %s\n", name.c_str());
  }
  auto found = cache_.find(name);
  if (found != cache_.end()) {
    if (!resident_) {
      ++cache_hits_;
      if (!shared_cache_) touch(name);
    }
    return found->second;
  }
  Tensor result = read(name, "data_offsets", false);
  if (resident_) {
    cache_[name] = result;
  } else if (shared_cache_) {
    retain_shared(name, result, false);
  } else if (cache_budget_ > 0) {
    admit(name, result);
  }
  return result;
}

void WeightStore::prefetch(const std::string& prefix, const std::vector<std::string>& suffixes) {
  if (cache_budget_ != 0 && !shared_cache_) throw std::logic_error("Weight prefetch requires streaming weights");
  const std::lock_guard lock(prefetch_mutex_);
  // Only the listed tensors are read. A block prefix also covers weights that another phase
  // owns, and reading those would hold memory and delay the tensors in use.
  for (const std::string& suffix : suffixes) {
    const std::string name = prefix + suffix;
    if (!has(name)) continue;
    const bool quantized = dtype(name).starts_with("Q4");
    for (const bool scales : {false, true}) {
      if (scales && !quantized) continue;
      const std::string key = scales ? name + "/scale" : name;
      if (cache_.count(key) || prefetch_pending_.count(key)) continue;
      uint64_t bytes = 0;
      if (shared_cache_) {
        bytes = read_range(name, scales ? "scale_offsets" : "data_offsets").second;
        if (bytes > cache_budget_ - shared_resident_plan_bytes_ - pending_prefetch_bytes_) {
          ++skipped_prefetch_requests_;
          continue;
        }
        pending_prefetch_bytes_ += bytes;
        peak_prefetch_bytes_ = std::max(peak_prefetch_bytes_, pending_prefetch_bytes_);
        peak_shared_bytes_ = std::max(peak_shared_bytes_, cache_bytes_ + pending_prefetch_bytes_);
      }
      prefetch_queue_.push_back({name, scales});
      prefetch_pending_.emplace(key, bytes);
    }
  }
  if (prefetch_queue_.empty()) return;
  while (prefetch_readers_.size() < reader_threads()) {
    prefetch_readers_.emplace_back([this] { read_loop(); });
  }
  work_available_.notify_all();
}

void WeightStore::finish_prefetch() {
  const auto began = std::chrono::steady_clock::now();
  std::unique_lock lock(prefetch_mutex_);
  result_ready_.wait(lock, [this] { return prefetch_queue_.empty() && prefetch_active_ == 0; });
  const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - began).count();
  prefetch_wait_ns_ += wait_ns;
  if (profile_foreground_reads_) prefetch_drain_ns_ += wait_ns;
  if (!prefetch_errors_.empty()) {
    std::exception_ptr error = prefetch_errors_.begin()->second;
    prefetch_errors_.erase(prefetch_errors_.begin());
    std::rethrow_exception(error);
  }
}

Tensor WeightStore::take_prefetched(const std::string& key) {
  std::unique_lock lock(prefetch_mutex_);
  const auto pending = prefetch_pending_.find(key);
  if (pending == prefetch_pending_.end()) return {};
  const auto began = std::chrono::steady_clock::now();
  const auto ready = [&] {
    return prefetch_stop_ || prefetched_.count(key) || prefetch_errors_.count(key);
  };
  const bool blocked = profile_foreground_reads_ && !ready();
  result_ready_.wait(lock, ready);
  const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - began).count();
  prefetch_wait_ns_ += wait_ns;
  if (blocked) {
    auto& read = foreground_read(key);
    ++read.blocked_calls;
    read.blocked_ns += wait_ns;
  }
  pending_prefetch_bytes_ -= pending->second;
  prefetch_pending_.erase(pending);
  if (auto found = prefetched_.find(key); found != prefetched_.end()) {
    Tensor value = found->second;
    prefetched_.erase(found);
    return value;
  }
  if (auto found = prefetch_errors_.find(key); found != prefetch_errors_.end()) {
    std::exception_ptr error = found->second;
    prefetch_errors_.erase(found);
    std::rethrow_exception(error);
  }
  return {};  // The pipeline stopped with this read unfinished; the caller reads synchronously.
}

void WeightStore::read_loop() {
  for (;;) {
    Request request;
    {
      std::unique_lock lock(prefetch_mutex_);
      work_available_.wait(lock, [this] { return prefetch_stop_ || !prefetch_queue_.empty(); });
      if (prefetch_stop_ && prefetch_queue_.empty()) return;
      request = std::move(prefetch_queue_.front());
      prefetch_queue_.pop_front();
      ++prefetch_active_;
    }
    std::exception_ptr error;
    Tensor value{};
    @autoreleasepool {
      try {
        value = request.scales ? read_scale(request.name, true) : read(request.name, "data_offsets", false, true);
      } catch (...) {
        error = std::current_exception();
      }
    }
    {
      const std::lock_guard lock(prefetch_mutex_);
      --prefetch_active_;
      const std::string key = request.scales ? request.name + "/scale" : request.name;
      if (error) {
        prefetch_errors_[key] = error;
      } else {
        prefetched_[key] = value;
      }
    }
    result_ready_.notify_all();
  }
}

std::vector<int64_t> WeightStore::shape(const std::string& name) const {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown tensor: " + name);
  return shape_of(spec[@"shape"]);
}

std::pair<uint64_t, uint64_t> WeightStore::read_range(const std::string& name, const std::string& field) const {
  NSDictionary* spec = specs_[[NSString stringWithUTF8String:name.c_str()]];
  if (!spec) throw std::runtime_error("Unknown tensor: " + name);
  NSArray* offsets = spec[[NSString stringWithUTF8String:field.c_str()]];
  if (!offsets) throw std::runtime_error("Missing tensor offsets: " + name);
  const uint64_t begin = [offsets[0] unsignedLongLongValue];
  const uint64_t end = [offsets[1] unsignedLongLongValue];
  if (end < begin) throw std::runtime_error("Invalid tensor offsets: " + name);
  return {begin, end - begin};
}

Tensor WeightStore::read(const std::string& name, const std::string& field, bool scales, bool background) {
  const auto [begin, length] = read_range(name, field);
  const auto read_start = std::chrono::steady_clock::now();
  id<MTLBuffer> buffer = [device_ newBufferWithLength:length options:resource_options];
  if (!buffer) throw std::runtime_error("Metal weight allocation failed: " + name);
  uint64_t completed = 0;
  while (completed < length) {
    const ssize_t result = pread(fd_, static_cast<uint8_t*>(buffer.contents) + completed, length - completed, data_begin_ + begin + completed);
    if (result < 0) throw std::runtime_error(std::strerror(errno));
    if (result == 0) throw std::runtime_error("Unexpected end of model file: " + name);
    completed += result;
  }
  uint64_t elapsed_ns;
  {
    const std::lock_guard lock(read_stats_mutex_);
    bytes_read_ += length;
    read_bytes_by_name_[name] += length;
    ++read_calls_;
    elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - read_start).count();
    read_ns_ += elapsed_ns;
  }
  if (!background && profile_foreground_reads_) {
    auto& profile = foreground_read(name);
    ++profile.direct_calls;
    profile.direct_bytes += length;
    profile.direct_ns += elapsed_ns;
  }
  const std::string type = scales ? "F32" : dtype(name);
  return Tensor{buffer, shape(name), data_type(type)};
}

Tensor WeightStore::scale(const std::string& name) {
  const std::string key = name + "/scale";
  auto found = cache_.find(key);
  if (found != cache_.end()) {
    if (shared_cache_) ++cache_hits_;
    return found->second;
  }
  Tensor result = take_prefetched(key);
  if (!result.buffer) result = read_scale(name);
  if (shared_cache_) {
    retain_shared(key, result, true);
  } else {
    // Legacy streaming and LRU retain scales outside their optional data-cache budget.
    cache_[key] = result;
    scale_cache_bytes_ += result.buffer.length;
  }
  return result;
}

Tensor WeightStore::read_scale(const std::string& name, bool background) {
  Tensor result = read(name, "scale_offsets", true, background);
  const uint64_t group = std::stoull(dtype(name).substr(3));
  result.shape = {result.shape[0], result.shape[1] / static_cast<int64_t>(group)};
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
  id<MTLBuffer> buffer = [device_ newBufferWithBytes:data.bytes length:data.length options:resource_options];
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
