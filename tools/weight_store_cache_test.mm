// Check the WeightStore LRU for bitwise consistency.
// Read the same pack four times, using each storage policy:
//   streaming with a zero cache budget
//   streaming through the background prefetch pipeline
//   bounded LRU that can evict entries
//   fully resident weights
// Compare tensor SHA-256 digests, cache budgets, and hit counts.
// An absent optional probe pack is reported as skipped.

#import "native.hpp"
#import <CommonCrypto/CommonDigest.h>

#include <array>
#include <cstdio>
#include <map>
#include <string>

namespace {

std::string sha256_hex(const onyx::Tensor& value) {
  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(value.data(), static_cast<CC_LONG>(value.buffer.length), digest);
  char hex[CC_SHA256_DIGEST_LENGTH * 2 + 1] = {0};
  for (size_t index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index) {
    std::snprintf(hex + index * 2, 3, "%02x", digest[index]);
  }
  return hex;
}

std::map<std::string, std::string> read_all(onyx::WeightStore& store, uint64_t* hits_delta_begin) {
  std::map<std::string, std::string> hashes;
  const auto names = store.names();
  for (int pass = 0; pass < 2; ++pass) {
    for (const std::string& name : names) {
      const std::string digest = sha256_hex(store.tensor(name));
      if (pass == 1) hashes[name] = digest;
    }
  }
  (void)hits_delta_begin;
  return hashes;
}

bool direct_profile_matches(const onyx::WeightStore& store, uint64_t calls, uint64_t bytes) {
  const auto json = store.foreground_read_profile_json();
  NSData* data = [NSData dataWithBytes:json.data() length:json.size()];
  NSDictionary* profile = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  if (!profile || !profile[@"groups"]) return false;
  uint64_t observed_calls = 0;
  uint64_t observed_bytes = 0;
  for (NSDictionary* group in [profile[@"groups"] allValues]) {
    observed_calls += [group[@"direct_calls"] unsignedLongLongValue];
    observed_bytes += [group[@"direct_bytes"] unsignedLongLongValue];
  }
  return observed_calls == calls && observed_bytes == bytes;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::fprintf(stderr, "usage: onyx-weight-store-cache-test <pack> [budget_bytes]\n");
      return 2;
    }
    const std::string pack = argv[1];
    FILE* probe = std::fopen(pack.c_str(), "rb");
    if (!probe) {
      std::printf("{\"test\":\"onyx_weight_store_cache\",\"status\":\"skip\",\"reason\":\"pack missing\"}\n");
      return 0;
    }
    std::fclose(probe);
    const uint64_t budget = argc > 2 ? std::stoull(argv[2]) : (64ull << 20);
    try {
      onyx::WeightStore streaming(pack, false, 0);
      streaming.enable_foreground_read_profiling();
      const auto streaming_hashes = read_all(streaming, nullptr);
      bool profile_ok = direct_profile_matches(streaming, streaming.read_calls(), streaming.bytes_read());

      // Queue each tensor one name at a time and take it right away, so the reader pipeline
      // hands back the same bytes as the synchronous path.
      std::map<std::string, std::string> pipeline_hashes;
      {
        onyx::WeightStore pipeline(pack, false, 0);
        pipeline.enable_foreground_read_profiling();
        for (const std::string& name : pipeline.names()) {
          pipeline.prefetch("", {name});
          pipeline_hashes[name] = sha256_hex(pipeline.tensor(name));
          if (pipeline.dtype(name).starts_with("Q4")) {
            pipeline_hashes[name + "/scale"] = sha256_hex(pipeline.scale(name));
          }
        }
        pipeline.finish_prefetch();
        profile_ok = profile_ok && direct_profile_matches(pipeline, 0, 0);
      }
      bool pipeline_ok = pipeline_hashes.size() >= streaming_hashes.size();
      for (const auto& [name, digest] : streaming_hashes) {
        pipeline_ok = pipeline_ok && pipeline_hashes.at(name) == digest;
        if (streaming.dtype(name).starts_with("Q4")) {
          pipeline_ok = pipeline_ok && pipeline_hashes.at(name + "/scale") == sha256_hex(streaming.scale(name));
        }
      }

      onyx::WeightStore lru(pack, false, budget);
      const auto lru_hashes = read_all(lru, nullptr);

      onyx::WeightStore resident(pack, true);
      const auto resident_hashes = read_all(resident, nullptr);

      if (streaming_hashes.empty() || streaming_hashes.size() != resident_hashes.size() || lru_hashes.size() != resident_hashes.size()) {
        std::printf("{\"test\":\"onyx_weight_store_cache\",\"status\":\"fail\",\"reason\":\"name sets differ\"}\n");
        return 1;
      }
      for (const auto& [name, digest] : resident_hashes) {
        if (streaming_hashes.at(name) != digest || lru_hashes.at(name) != digest) {
          std::printf("{\"test\":\"onyx_weight_store_cache\",\"status\":\"fail\",\"reason\":\"hash mismatch\",\"tensor\":\"%s\"}\n", name.c_str());
          return 1;
        }
      }
      const bool budget_ok = lru.cache_bytes() <= budget;
      // Oversized tensors are ineligible. A populated cache should get hits on the second pass.
      const bool hits_ok = lru.cache_entries() == 0 || lru.cache_hits() > 0;
      const bool cached_ok = resident_hashes.size() > 0;
      const bool ok = pipeline_ok && budget_ok && hits_ok && cached_ok && profile_ok;
      std::printf("{\"test\":\"onyx_weight_store_cache\",\"status\":\"%s\",\"tensors\":%zu,\"budget\":%llu,"
                  "\"cache_bytes\":%llu,\"entries\":%llu,\"hits\":%llu,\"prefetch_pipeline\":\"%s\","
                  "\"foreground_read_profile\":\"%s\"}\n",
                  ok ? "pass" : "fail",
                  resident_hashes.size(),
                  static_cast<unsigned long long>(lru.cache_budget()),
                  static_cast<unsigned long long>(lru.cache_bytes()),
                  static_cast<unsigned long long>(lru.cache_entries()),
                  static_cast<unsigned long long>(lru.cache_hits()),
                  pipeline_ok ? "pass" : "fail", profile_ok ? "pass" : "fail");
      return ok ? 0 : 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "onyx-weight-store-cache-test: %s\n", error.what());
      return 1;
    }
  }
}
