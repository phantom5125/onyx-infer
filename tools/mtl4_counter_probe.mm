// Probe Metal 4 GPU timestamps for individual BF16 Q4 GEMM dispatches.
// Usage: mtl4_counter_probe <metallib> <onxpack> [m]
#import <Metal/Metal.h>
#import <Metal/MTL4Counters.h>
#import <Metal/MTL4CommandBuffer.h>
#import <Metal/MTL4CommandQueue.h>
#import <Metal/MTL4ArgumentTable.h>
#import <Metal/MTL4ComputeCommandEncoder.h>
#import <Foundation/Foundation.h>
#import <objc/message.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "native.hpp"

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) { std::printf("usage: <metallib> <onxpack> [m]\n"); return 2; }
    const uint32_t m = argc > 3 ? uint32_t(std::atoi(argv[3])) : 1040;
    const std::string prefix = "blocks.0.attn.qkv_proj";
    NSError* error = nil;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    onyx::WeightStore store(argv[2]);
    onyx::Tensor weight = store.tensor(prefix + ".weight");
    onyx::Tensor scales_t = store.scale(prefix + ".weight");
    const uint32_t k = uint32_t(store.shape(prefix + ".weight")[1]);
    const uint32_t n = uint32_t(store.shape(prefix + ".weight")[0]);

    // Generate BF16 activations.
    id<MTLBuffer> x_buf = [device newBufferWithLength:size_t(m) * k * 2 options:MTLResourceStorageModeShared];
    {
      auto* x16 = static_cast<uint16_t*>(x_buf.contents);
      unsigned st = 7;
      for (size_t i = 0; i < size_t(m) * k; ++i) {
        st = st * 1664525u + 1013904223u;
        const float v = float(int(st >> 16) % 2000 - 1000) / 1000.0f;
        uint32_t bits; std::memcpy(&bits, &v, 4);
        x16[i] = uint16_t(bits >> 16);
      }
    }
    id<MTLBuffer> y_buf = [device newBufferWithLength:size_t(m) * n * 2 options:MTLResourceStorageModeShared];

    // bf16 GEMM pipeline
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
    if (!library) { std::printf("library load failed\n"); return 1; }
    id<MTLFunction> gemm_fn = [library newFunctionWithName:@"onyx_q4_gemm_simd_bn32_bk32_bf16"];
    if (!gemm_fn) { std::printf("kernel not found\n"); return 1; }
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:gemm_fn error:&error];
    if (!ps) { std::printf("pipeline failed\n"); return 1; }

    // Create the Metal 4 queue and timestamp heap.
    id<MTL4CommandQueue> q4 = [device newMTL4CommandQueue];
    if (!q4) { std::printf("MTL4CommandQueue failed\n"); return 1; }
    MTL4CounterHeapDescriptor* hd = [MTL4CounterHeapDescriptor new];
    hd.type = MTL4CounterHeapTypeTimestamp;
    hd.count = 8;
    id heap_id = ((id(*)(id, SEL, id))objc_msgSend)(device, sel_registerName("newCounterHeapWithDescriptor:"), hd);  // Resolve the selector dynamically for SDKs without the declaration.
    if (!heap_id) { std::printf("counter heap failed\n"); return 1; }
    id<MTL4CounterHeap> heap = heap_id;
    if (!heap) { std::printf("counter heap failed\n"); return 1; }

    // Bindings: activation, weights, scales, output, four constants, zero bias.
    MTL4ArgumentTableDescriptor* ad = [MTL4ArgumentTableDescriptor new];
    ad.maxBufferBindCount = 16;
    id table_id = ((id(*)(id, SEL, id, id))objc_msgSend)(device, sel_registerName("newMTL4ArgumentTableWithDescriptor:error:"), ad, error);
    if (!table_id) { std::printf("argument table failed: %s\n", error.localizedDescription.UTF8String); return 1; }
    id<MTL4ArgumentTable> table = table_id;

    int32_t consts[4] = {int32_t(m), int32_t(n), int32_t(k), 64};  // m, n, k, group → index 4..7
    id<MTLBuffer> consts_buf = [device newBufferWithBytes:consts length:16 options:MTLResourceStorageModeShared];
    id<MTLBuffer> zero_bias = [device newBufferWithLength:size_t(n) * 2 options:MTLResourceStorageModeShared];

    [table setAddress:x_buf.gpuAddress atIndex:0];
    [table setAddress:weight.buffer.gpuAddress atIndex:1];
    [table setAddress:scales_t.buffer.gpuAddress atIndex:2];
    [table setAddress:y_buf.gpuAddress atIndex:3];
    [table setAddress:consts_buf.gpuAddress atIndex:4];
    [table setAddress:zero_bias.gpuAddress atIndex:8];

    // Run one warmup followed by one measurement.
    for (int iter = 0; iter < 2; ++iter) {
      dispatch_semaphore_t done = dispatch_semaphore_create(0);
      id cb_id = ((id(*)(id, SEL))objc_msgSend)(q4, sel_registerName("commandBuffer"));
      id<MTL4CommandBuffer> cb = cb_id;
      id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:ps];
      [enc setArgumentTable:table];
      const NSUInteger base = 2 * iter;
      ((void(*)(id, SEL, id, unsigned long))objc_msgSend)(enc, sel_registerName("writeTimestampIntoHeap:atIndex:"), heap, base);
      [enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, (m + 31) / 32, 1)
           threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
      ((void(*)(id, SEL, id, unsigned long))objc_msgSend)(enc, sel_registerName("writeTimestampIntoHeap:atIndex:"), heap, base + 1);
      [enc endEncoding];
      id handler = ^(id<MTL4CommandBuffer>) { dispatch_semaphore_signal(done); };
      ((void(*)(id, SEL, id))objc_msgSend)(cb, sel_registerName("addCompletedHandler:"), handler);
      id<MTL4CommandBuffer> cbs[1] = {cb};
      [q4 commit:cbs count:1];
      dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    }
    NSData* resolved = [heap resolveCounterRange:NSMakeRange(0, 4)];
    if (!resolved) { std::printf("resolve failed\n"); return 1; }
    uint64_t ticks[4] = {0, 0, 0, 0};
    [resolved getBytes:ticks length:sizeof(ticks)];
    // Convert GPU ticks to nanoseconds with paired device timestamps.
    MTLTimestamp cpu0, gpu0, cpu1, gpu1;
    [device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    [device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    const double ns_per_tick = double(cpu1 - cpu0) / double(gpu1 - gpu0);
    for (int iter = 0; iter < 2; ++iter) {
      const double ns = double(ticks[2 * iter + 1] - ticks[2 * iter]) * ns_per_tick;
      std::printf("{\"iter\":%d,\"gpu_ms\":%.3f}\n", iter, ns / 1e6);
    }
    return 0;
  }
}
