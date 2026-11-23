#import "native.hpp"

#include <stdexcept>
#include <string>
#include <vector>

// Q4 linear layers on the M5 neural accelerators.
//
// The kernel is MLX's affine_qmm_t_nax, compiled from the vendored source into the Onyx
// metallib by ops/metal/q4_gemm_nax.metal. It dequantizes inside the matrix product through
// Metal Performance Primitives, where the MPSGraph path materializes a dequantized weight.
namespace onyx {
namespace {

// MLX kernel names for BF16 activations, group size 64, and 4 bits.
constexpr const char* kQmmName = "affine_qmm_t_nax_bfloat16_t_gs_64_b_4_hgs_0_alN_";

id<MTLLibrary> nax_library(id<MTLDevice> device) {
  static id<MTLLibrary> library;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    NSError* error = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:ONYX_Q4_METALLIB]];
    library = [device newLibraryWithURL:url error:&error];
    if (!library) throw std::runtime_error(std::string("Could not load the q4 kernel library: ") + error.localizedDescription.UTF8String);
  });
  return library;
}

}  // namespace

bool GraphSession::nax_available() {
  // Apple10 is the M5 GPU family, the first with neural accelerators. MLX gates the same
  // kernels on the M5 generation.
  static const bool available = [queue_.device supportsFamily:MTLGPUFamilyApple10];
  return available;
}

Tensor GraphSession::linear_q4_nax(const std::string& weight_name, const Tensor& weight, const Tensor& scales,
                                   const Tensor& x, int64_t n, int64_t k) {
  const uint32_t rows = uint32_t(x.elements() / k);
  const bool aligned = n % 64 == 0;
  // The staged kernel reads activations through aligned vector loads.
  const bool staged = aligned && x.byte_offset % 8 == 0;
  const uint32_t block_rows = staged || rows > 32 ? 64 : 32;
  const std::string name = std::string(kQmmName) + (aligned ? "1" : "0") + "_batch_0_bm_" + std::to_string(block_rows) +
                           (staged ? "_bn_64_wm_4_wn_1" : "_bn_64_wm_2_wn_2");
  id<MTLComputePipelineState> pipeline = nax_pipelines_[name];
  if (!pipeline) {
    NSError* error = nil;
    id<MTLFunction> function = [nax_library(queue_.device) newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!function) throw std::runtime_error("NAX qmm kernel not found in the q4 kernel library: " + name);
    pipeline = [queue_.device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) throw std::runtime_error(std::string("NAX qmm pipeline failed: ") + error.localizedDescription.UTF8String);
    nax_pipelines_[name] = pipeline;
  }

  // Onyx stores (nibble - 8) * scale and the kernel computes nibble * scale + bias, so each
  // affine bias is -8 times its scale. Like MLX, the kernel takes both in the activation type.
  auto affine = nax_affine_.find(weight_name);
  if (affine == nax_affine_.end()) {
    const uint64_t count = scales.elements();
    std::vector<uint16_t> scale_bf16(count);
    std::vector<uint16_t> bias_bf16(count);
    const float* values = static_cast<const float*>(scales.data());
    for (uint64_t index = 0; index < count; ++index) {
      scale_bf16[index] = float_to_bf16(values[index]);
      bias_bf16[index] = float_to_bf16(-8.0f * values[index]);
    }
    id<MTLDevice> device = queue_.device;
    Tensor scale{[device newBufferWithBytes:scale_bf16.data() length:count * 2 options:MTLResourceStorageModeShared],
                 scales.shape, MPSDataTypeBFloat16};
    Tensor bias{[device newBufferWithBytes:bias_bf16.data() length:count * 2 options:MTLResourceStorageModeShared],
                scales.shape, MPSDataTypeBFloat16};
    if (!scale.buffer || !bias.buffer) throw std::runtime_error("NAX qmm scale allocation failed: " + weight_name);
    affine = nax_affine_.emplace(weight_name, std::pair<Tensor, Tensor>{scale, bias}).first;
  }

  Tensor result{nil, {rows, n}, MPSDataTypeBFloat16};
  result.buffer = [queue_.device newBufferWithLength:result.byte_size() options:MTLResourceStorageModeShared];
  if (!result.buffer) throw std::runtime_error("NAX qmm output allocation failed: " + weight_name);
  id<MTLCommandBuffer> command = [queue_ commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  // Bindings 0-7 follow MLX: weights, scales, biases, activations, output, K, N, M.
  [encoder setBuffer:weight.buffer offset:weight.byte_offset atIndex:0];
  [encoder setBuffer:affine->second.first.buffer offset:0 atIndex:1];
  [encoder setBuffer:affine->second.second.buffer offset:0 atIndex:2];
  [encoder setBuffer:x.buffer offset:x.byte_offset atIndex:3];
  [encoder setBuffer:result.buffer offset:0 atIndex:4];
  const int32_t k_int = int32_t(k), n_int = int32_t(n), m_int = int32_t(rows);
  [encoder setBytes:&k_int length:4 atIndex:5];
  [encoder setBytes:&n_int length:4 atIndex:6];
  [encoder setBytes:&m_int length:4 atIndex:7];
  const MTLSize groups = MTLSizeMake((n + 63) / 64, (rows + block_rows - 1) / block_rows, 1);
  [encoder dispatchThreadgroups:groups threadsPerThreadgroup:staged ? MTLSizeMake(128, 1, 1) : MTLSizeMake(32, 2, 2)];
  [encoder endEncoding];
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    throw std::runtime_error(std::string("NAX qmm failed: ") + command.error.localizedDescription.UTF8String);
  }
  return result;
}

}  // namespace onyx
