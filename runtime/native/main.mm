#import "native.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

bool flag(int argc, char** argv, std::string_view name) {
  return std::any_of(argv + 2, argv + argc, [name](std::string_view arg) { return arg == name; });
}

// Weight prefetch is explicit: --no-prefetch-weights wins when both flags are present.
bool weight_prefetch(int argc, char** argv) {
  return flag(argc, argv, "--prefetch-weights") && !flag(argc, argv, "--no-prefetch-weights");
}

std::string argument(int argc, char** argv, const std::string& name, bool required = true) {
  for (int index = 2; index < argc; ++index) {
    if (name == argv[index]) {
      if (index + 1 == argc) throw std::runtime_error("Missing value for " + name);
      return argv[index + 1];
    }
  }
  if (required) throw std::runtime_error("Missing required argument: " + name);
  return {};
}

// Return the supplied default when the optional argument is absent.
uint64_t optional_bytes(int argc, char** argv, const std::string& name) {
  for (int index = 2; index + 1 < argc; ++index) {
    if (name == argv[index]) {
      try {
        return std::stoull(argv[index + 1]);
      } catch (const std::invalid_argument&) {
        throw std::runtime_error("Invalid value for " + name + ": " + argv[index + 1]);
      } catch (const std::out_of_range&) {
        throw std::runtime_error("Value out of range for " + name + ": " + argv[index + 1]);
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    try {
      if (argc < 2) throw std::runtime_error("Expected a native stage: text");
      const std::string stage = argv[1];
      const onyx::StageOptions options = onyx::parse_stage_options(argc, argv);
      if (stage == "text") {
        onyx::StageOptions text_options = options;
        text_options.prefetch_weights = weight_prefetch(argc, argv);
        onyx::encode_text(argument(argc, argv, "--model"), argument(argc, argv, "--tokens"), argument(argc, argv, "--output"), text_options);
        return 0;
      }
      if (stage == "dit") {
        const onyx::StageOptions dit_options = [&] {
          onyx::StageOptions copy = options;
          copy.cache_bytes = optional_bytes(argc, argv, "--cache-bytes");
          copy.modulation_cache_dir = argument(argc, argv, "--modulation-cache", false);
          copy.prefetch_weights = weight_prefetch(argc, argv);
          return copy;
        }();
        onyx::denoise_image(
            argument(argc, argv, "--model"), argument(argc, argv, "--embedding"),
            argument(argc, argv, "--noise"), argument(argc, argv, "--pdd"),
            argument(argc, argv, "--output"), std::stoul(argument(argc, argv, "--width")),
            std::stoul(argument(argc, argv, "--height")), dit_options);
        return 0;
      }
      if (stage == "vae") {
        onyx::decode_image(argument(argc, argv, "--model"), argument(argc, argv, "--latent"), argument(argc, argv, "--config"), argument(argc, argv, "--output"), options);
        return 0;
      }
      throw std::runtime_error("Unknown native stage: " + stage);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "onyx-native: %s\n", error.what());
      return 1;
    }
  }
}
