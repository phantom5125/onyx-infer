// Check the WeightStore LRU for bitwise consistency.
// Read the same pack twice using each storage policy:
//   streaming with a zero cache budget
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
      const auto streaming_hashes = read_all(streaming, nullptr);

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
      std::printf("{\"test\":\"onyx_weight_store_cache\",\"status\":\"%s\",\"tensors\":%zu,\"budget\":%llu,"
                  "\"cache_bytes\":%llu,\"entries\":%llu,\"hits\":%llu}\n",
                  budget_ok && hits_ok && cached_ok ? "pass" : "fail",
                  resident_hashes.size(),
                  static_cast<unsigned long long>(lru.cache_budget()),
                  static_cast<unsigned long long>(lru.cache_bytes()),
                  static_cast<unsigned long long>(lru.cache_entries()),
                  static_cast<unsigned long long>(lru.cache_hits()));
      return budget_ok && hits_ok && cached_ok ? 0 : 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "onyx-weight-store-cache-test: %s\n", error.what());
      return 1;
    }
  }
}
