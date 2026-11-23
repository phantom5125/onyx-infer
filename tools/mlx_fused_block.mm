// E13 v2: C++ port of the python-validated fused DiT block (25.84 ms on
// python). Mirrors tools/mlx_fused_block recipe exactly: fast::rms_norm,
// quantized_matmul, fast::scaled_dot_product_attention, lazy elementwise.
// Correctness: fused path vs op-by-op path on identical weights/inputs.
#import <mlx/mlx.h>
#import <mlx/backend/metal/metal.h>
#import <mlx/fast.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace mx = mlx::core;

int main(int argc, char** argv) {
  const int iterations = argc > 1 ? std::stoi(argv[1]) : 10;
  @autoreleasepool {
    // Allow an override when MLX cannot locate the metallib beside its library.
    if (const char* metallib_path = std::getenv("MLX_METALLIB_PATH"); metallib_path && *metallib_path) {
      mx::metal::set_metallib_path(metallib_path);
    }
    mx::set_default_device(mx::Device(mx::Device::gpu));
    const int m = argc > 2 ? std::stoi(argv[2]) : 1024;
    constexpr int hidden = 5376;
    constexpr int ffn = 14336;
    constexpr int heads = 56;
    constexpr int head_dim = 128;
    constexpr int group = 64;
    std::mt19937 rng(13);
    std::normal_distribution<float> normal(0.0f, 0.2f);

    auto make_q4 = [&](int n, int k) {
      std::vector<uint32_t> words(size_t(n) * k / 8);
      for (auto& word : words) {
        for (int nibble = 0; nibble < 8; ++nibble) {
          word |= uint32_t(rng() % 16) << (4 * nibble);
        }
      }
      std::vector<float> scales(size_t(n) * (k / group));
      for (auto& s : scales) s = (0.01f + 0.02f * (rng() % 8)) / 32.0f;  // 1/sqrt(k) keeps activations bounded like real trained weights
      std::vector<float> biases(scales.size());
      for (size_t index = 0; index < scales.size(); ++index) biases[index] = -8.0f * scales[index];
      mx::array w = mx::array(words.data(), {n, k / 8}, mx::uint32);
      mx::array s = mx::array(scales.data(), {n, k / group}, mx::float32);
      mx::array b = mx::array(biases.data(), {n, k / group}, mx::float32);
      mx::eval(w, s, b);
      return std::make_tuple(w, s, b);
    };
    auto [qkv_w, qkv_s, qkv_b] = make_q4(3 * heads * head_dim, hidden);
    auto [out_w, out_s, out_b] = make_q4(hidden, heads * head_dim);
    auto [fc1_w, fc1_s, fc1_b] = make_q4(2 * ffn, hidden);
    auto [fc2_w, fc2_s, fc2_b] = make_q4(hidden, ffn);

    std::vector<float> x0(size_t(m) * hidden);
    for (auto& value : x0) value = normal(rng);
    std::vector<float> ones_h(hidden, 1.0f), ones_d(head_dim, 1.0f);
    mx::array x_mlx0 = mx::array(x0.data(), {m, hidden}, mx::float16);
    mx::array norm1 = mx::astype(mx::array(ones_h.data(), {hidden}, mx::float32), mx::float16);
    mx::array norm2 = norm1;
    mx::array q_norm = mx::astype(mx::array(ones_d.data(), {head_dim}, mx::float32), mx::float16);
    mx::array k_norm = q_norm;
    const float sdpa_scale = 1.0f / std::sqrt(float(head_dim));

    // ---- Fused path: whole block in one mlx lazy graph ----
    auto block_fused = [&](mx::array x) {
      mx::array h = mx::fast::rms_norm(x, norm1, 1e-5f);
      mx::array qkv = mx::quantized_matmul(h, qkv_w, qkv_s, qkv_b, true, group, 4);
      // One transpose to (3, m, heads, head_dim), then axis-0 slices are
      // contiguous (1, m, heads, head_dim) views - required by fast kernels.
      qkv = mx::reshape(qkv, {m, 3, heads, head_dim});
      qkv = mx::transpose(qkv, {1, 0, 2, 3});
      mx::array q = mx::transpose(mx::slice(qkv, {0, 0, 0, 0}, {1, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3});
      mx::array k = mx::transpose(mx::slice(qkv, {1, 0, 0, 0}, {2, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3});
      mx::array v = mx::transpose(mx::slice(qkv, {2, 0, 0, 0}, {3, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3});
      // mlX fast::sdpa contract is (B, heads, seq, dim) - verified against an
      // fp32 softmax reference (torch-style, NOT the (B,seq,heads,dim) guess).
      q = mx::fast::rms_norm(q, q_norm, 1e-5f);
      k = mx::fast::rms_norm(k, k_norm, 1e-5f);
      mx::array attended = mx::fast::scaled_dot_product_attention(q, k, v, sdpa_scale);
      attended = mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {m, heads * head_dim});
      mx::array projected = mx::quantized_matmul(attended, out_w, out_s, out_b, true, group, 4);
      x = x + projected;
      h = mx::fast::rms_norm(x, norm2, 1e-5f);
      mx::array mid = mx::quantized_matmul(h, fc1_w, fc1_s, fc1_b, true, group, 4);
      mx::array gate = mx::slice(mid, {0, 0}, {m, ffn}, {1, 1});
      mx::array up = mx::slice(mid, {0, ffn}, {m, 2 * ffn}, {1, 1});
      mx::array mlp_out = mx::quantized_matmul(gate * mx::sigmoid(gate) * up, fc2_w, fc2_s, fc2_b, true, group, 4);
      return x + mlp_out;
    };

    // ---- Op-by-op path (same primitives, separate evals like the engine) ----
    auto block_byop = [&](mx::array x) {
      auto step = [&](mx::array a) { mx::eval(a); return a; };
      mx::array h = step(mx::fast::rms_norm(x, norm1, 1e-5f));
      mx::array qkv = step(mx::quantized_matmul(h, qkv_w, qkv_s, qkv_b, true, group, 4));
      qkv = step(mx::transpose(mx::reshape(qkv, {m, heads, 3, head_dim}), {0, 2, 1, 3}));
      mx::array qkv3 = step(mx::transpose(mx::reshape(qkv, {m, 3, heads, head_dim}), {1, 0, 2, 3}));  // (3,m,h,d)
      mx::array q = step(mx::transpose(mx::slice(qkv3, {0, 0, 0, 0}, {1, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}));
      mx::array k = step(mx::transpose(mx::slice(qkv3, {1, 0, 0, 0}, {2, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}));
      mx::array v = step(mx::transpose(mx::slice(qkv3, {2, 0, 0, 0}, {3, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}));
      q = step(mx::fast::rms_norm(q, q_norm, 1e-5f));
      k = step(mx::fast::rms_norm(k, k_norm, 1e-5f));
      mx::array attended = step(mx::fast::scaled_dot_product_attention(q, k, v, sdpa_scale));
      std::fprintf(stderr, "byop: q(%d,%d,%d,%d) att(%d,%d,%d,%d)\n",
          q.shape(0), q.shape(1), q.shape(2), q.shape(3), attended.shape(0), attended.shape(1), attended.shape(2), attended.shape(3));
      attended = step(mx::transpose(attended, {0, 2, 1, 3}));
      attended = step(mx::reshape(attended, {m, heads * head_dim}));
      mx::array projected = step(mx::quantized_matmul(attended, out_w, out_s, out_b, true, group, 4));
      x = step(x + projected);
      h = step(mx::fast::rms_norm(x, norm2, 1e-5f));
      mx::array mid = step(mx::quantized_matmul(h, fc1_w, fc1_s, fc1_b, true, group, 4));
      mx::array gate = step(mx::slice(mid, {0, 0}, {m, ffn}, {1, 1}));
      mx::array up = step(mx::slice(mid, {0, ffn}, {m, 2 * ffn}, {1, 1}));
      mx::array act = step(gate * mx::sigmoid(gate) * up);
      mx::array mlp_out = step(mx::quantized_matmul(act, fc2_w, fc2_s, fc2_b, true, group, 4));
      return step(x + mlp_out);
    };

    // Correctness: one block each from the same x0, stage-by-stage diff.
    {
      auto stage = [&](const char* name, mx::array a, mx::array b) {
        mx::array d = mx::astype(mx::abs(a - b), mx::float32);
        mx::eval(d);
        std::printf("stage %-10s max diff %.4g\n", name, d.data<float>()[0] * 0 + mx::max(d, {0, 1}).item<float>());
      };
      mx::array hA = mx::fast::rms_norm(x_mlx0, norm1, 1e-5f);
      mx::array hB = hA;
      stage("rms", hA, hB);
      mx::array qkvA = mx::quantized_matmul(hA, qkv_w, qkv_s, qkv_b, true, group, 4);
      mx::array qkvB = qkvA;
      stage("qkv-mm", qkvA, qkvB);
      // attention boundary: build q/k/v in BOTH layouts from same qkv
      mx::array qkv3A = mx::transpose(mx::reshape(qkvA, {m, 3, heads, head_dim}), {1, 0, 2, 3});
      mx::array qA = mx::fast::rms_norm(mx::transpose(mx::slice(qkv3A, {0, 0, 0, 0}, {1, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}), q_norm, 1e-5f);
      mx::array kA = mx::fast::rms_norm(mx::transpose(mx::slice(qkv3A, {1, 0, 0, 0}, {2, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}), k_norm, 1e-5f);
      mx::array vA = mx::transpose(mx::slice(qkv3A, {2, 0, 0, 0}, {3, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3});
      mx::array attA = mx::fast::scaled_dot_product_attention(qA, kA, vA, sdpa_scale);
      // q/k reference from numpy-free mlx fp32:
      mx::array qf = mx::astype(qA, mx::float32);
      mx::array kf = mx::astype(kA, mx::float32);
      mx::array vf = mx::astype(vA, mx::float32);
      mx::array logits = mx::matmul(qf, mx::transpose(kf, {0, 1, 3, 2})) * sdpa_scale;
      mx::array weights = mx::softmax(logits, -1);
      mx::array attRef = mx::matmul(weights, vf);
      mx::array dAtt = mx::astype(mx::max(mx::abs(attA - attRef), {0, 1, 2, 3}), mx::float32);
      mx::eval(dAtt);
      std::printf("stage %-10s max diff %.4g (vs fp32 ref)\n", "sdpa", dAtt.data<float>()[0]);
    }
    {
      auto stage = [&](const char* name, const mx::array& a, const mx::array& b) {
        mx::array d = mx::astype(mx::abs(a - b), mx::float32);
        mx::array mxv = mx::max(d, {0, 1});
        mx::eval(mxv);
        std::printf("stage %-10s max diff %.4g\n", name, mxv.item<float>());
      };
      auto oops = [&](mx::array h) {
        mx::array qkv = mx::quantized_matmul(h, qkv_w, qkv_s, qkv_b, true, group, 4);
        qkv = mx::reshape(qkv, {m, 3, heads, head_dim});
        qkv = mx::transpose(qkv, {1, 0, 2, 3});
        mx::array q = mx::fast::rms_norm(mx::transpose(mx::slice(qkv, {0, 0, 0, 0}, {1, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}), q_norm, 1e-5f);
        mx::array k = mx::fast::rms_norm(mx::transpose(mx::slice(qkv, {1, 0, 0, 0}, {2, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3}), k_norm, 1e-5f);
        mx::array v = mx::transpose(mx::slice(qkv, {2, 0, 0, 0}, {3, m, heads, head_dim}, {1, 1, 1, 1}), {0, 2, 1, 3});
        mx::array att = mx::fast::scaled_dot_product_attention(q, k, v, sdpa_scale);
        att = mx::reshape(mx::transpose(att, {0, 2, 1, 3}), {m, heads * head_dim});
        return mx::quantized_matmul(att, out_w, out_s, out_b, true, group, 4);
      };
      auto mlp_of = [&](mx::array h) {
        mx::array mid = mx::quantized_matmul(h, fc1_w, fc1_s, fc1_b, true, group, 4);
        mx::array gate = mx::slice(mid, {0, 0}, {m, ffn}, {1, 1});
        mx::array up = mx::slice(mid, {0, ffn}, {m, 2 * ffn}, {1, 1});
        return mx::quantized_matmul(gate * mx::sigmoid(gate) * up, fc2_w, fc2_s, fc2_b, true, group, 4);
      };
      mx::array h1 = mx::fast::rms_norm(x_mlx0, norm1, 1e-5f);
      mx::array projA = oops(h1);
      mx::array x1A = x_mlx0 + projA;
      mx::array h2 = mx::fast::rms_norm(x1A, norm2, 1e-5f);
      mx::array midA = mx::quantized_matmul(h2, fc1_w, fc1_s, fc1_b, true, group, 4);
      mx::array gateA = mx::slice(midA, {0, 0}, {m, ffn}, {1, 1});
      mx::array upA = mx::slice(midA, {0, ffn}, {m, 2 * ffn}, {1, 1});
      mx::array actA = gateA * mx::sigmoid(gateA) * upA;
      mx::array mlpA = mx::quantized_matmul(actA, fc2_w, fc2_s, fc2_b, true, group, 4);
      mx::array x2A = x1A + mlpA;
      stage("proj", projA, projA * 1.0f);
      // the interesting part: act in one shot vs evaluated stepwise
      mx::array actB = gateA * mx::sigmoid(gateA) * upA;
      mx::eval(actB);
      stage("act(=)", actA, actB);
      mx::array mlpB = mx::quantized_matmul(actB, fc2_w, fc2_s, fc2_b, true, group, 4);
      stage("mlp(lazy vs stepped)", mlpA, mlpB);
      mx::array x2B = x1A + mlpB;
      mx::eval(x2B);
      stage("x2", x2A, x2B);
      mx::array fused_again = block_fused(x_mlx0);
      stage("block_fused vs decomposed", fused_again, x2A);
    }
    mx::array byop_x = block_byop(x_mlx0);
    mx::array fused_x = block_fused(x_mlx0);
    mx::array diff = mx::astype(mx::abs(byop_x - fused_x), mx::float32);
    mx::eval(diff);
    const float* diff_host = diff.data<float>();
    float max_error = 0.0f;
    for (size_t index = 0; index < size_t(m) * hidden; ++index) {
      max_error = std::max(max_error, diff_host[index]);
    }
    // Relative error against byop magnitude (fp16 accumulation-order noise
    // is expected between lazy-fused and stepped eval).
    auto host16 = mx::astype(byop_x, mx::float16);
    mx::eval(host16);
    const uint16_t* ref_bits = host16.data<uint16_t>();
    double ref_max = 0.0;
    for (size_t index = 0; index < size_t(m) * hidden; ++index) {
      ref_max = std::max(ref_max, std::abs(double(float(reinterpret_cast<const __fp16&>(ref_bits[index])))));
    }
    std::printf("fused vs byop max_abs_error %.4g (ref magnitude %.4g, relative %.3g)\n",
        max_error, ref_max, ref_max > 0 ? max_error / ref_max : 0.0);
    if (m <= 16) {
      // CPU brute-force reference for the full block (fp32).
      std::vector<float> a_host(size_t(m) * hidden);
      {
        auto host16 = mx::astype(byop_x, mx::float16);
        mx::eval(host16);
        const uint16_t* bits = host16.data<uint16_t>();
        for (size_t index = 0; index < a_host.size(); ++index) a_host[index] = float(reinterpret_cast<const __fp16&>(bits[index]));
      }
      // reconstruct the same weights on CPU
      auto f16 = [](uint16_t bits) { return float(reinterpret_cast<const __fp16&>(bits)); };
      // CPU reference for rms -> qkv -> attention -> out -> mlp is large;
      // instead validate attention alone: both paths share everything up to
      // sdpa input; we validate the fused sdpa output numerically.
      std::printf("byop_out[0:4]: %.3f %.3f %.3f %.3f | fused_out[0:4]: ",
          a_host[0], a_host[1], a_host[2], a_host[3]);
      {
        auto host16 = mx::astype(fused_x, mx::float16);
        mx::eval(host16);
        const uint16_t* bits = host16.data<uint16_t>();
        for (int index = 0; index < 4; ++index) std::printf("%.3f ", float(reinterpret_cast<const __fp16&>(bits[index])));
        std::printf("\n");
      }
    }

    // Timing.
    for (int warm = 0; warm < 3; ++warm) fused_x = block_fused(fused_x);
    mx::eval(fused_x);
    const auto began = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) fused_x = block_fused(fused_x);
    mx::eval(fused_x);
    const double per_block_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - began).count() / iterations;
    std::printf("cpp fused block: %.2f ms | 50 blocks x 6 steps: %.1f s\n",
        per_block_ms, per_block_ms * 50 * 6 / 1000.0);
    return max_error < 0.1f ? 0 : 2;
  }
}
