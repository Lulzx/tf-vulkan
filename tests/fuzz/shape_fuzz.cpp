// Randomized shape fuzz vs CPU reference (spec §15 fuzz): deterministic seed,
// non-multiples of workgroup size, transpose variants.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/stream.h"

static int failures = 0;
int main() {
#ifdef TFVK_HAS_VULKAN
  tfvk::VulkanDevice dev;
  if (!dev.Init("").ok()) { printf("SKIP Tier C\n"); return 0; }
  uint64_t heap = dev.info().heap_sizes.empty() ? 0 : dev.info().heap_sizes[0];
  tfvk::Allocator alloc(&dev, tfvk::ComputeBudgetBytes(heap, dev.info().memory_budget_ext, dev.info().budget_heap_size));
  tfvk::Stream stream(&dev, &alloc);
  tfvk::VulkanKernels k(&dev, &stream);
  if (!k.Init().ok()) { printf("SKIP kernels unavailable\n"); return 0; }
  std::mt19937 rng(42);
  std::normal_distribution<float> nd;
  for (int iter = 0; iter < 60; ++iter) {
    int64_t M = 1 + rng() % 65, K = 1 + rng() % 65, N = 1 + rng() % 65;
    bool ta = rng() % 2, tb = rng() % 2;
    std::vector<float> av(M * K), bv(K * N);
    for (auto& x : av) x = nd(rng);
    for (auto& x : bv) x = nd(rng);
    auto mk = [&](const std::vector<float>& v, bool transposed, int64_t r, int64_t c) {
      std::vector<float> stored = v;
      if (transposed) {  // store transposed layout
        stored.assign(v.size(), 0);
        for (int64_t i = 0; i < r; ++i) for (int64_t j = 0; j < c; ++j) stored[j * r + i] = v[i * c + j];
      }
      auto r_ = alloc.Allocate(stored.size() * 4, stream.completed_token());
      if (!r_.ok()) return (tfvk::Allocation*)nullptr;
      stream.MemcpyHtod(stored.data(), r_.value, 0, stored.size() * 4, false);
      return r_.value;
    };
    auto* a = mk(av, ta, M, K);
    auto* b = mk(bv, tb, K, N);
    auto cr = alloc.Allocate(M * N * 4, stream.completed_token());
    if (!a || !b || !cr.ok()) { printf("alloc fail\n"); return 1; }
    if (!k.MatMul(a, b, cr.value, (uint32_t)M, (uint32_t)N, (uint32_t)K, ta, tb).ok()) {
      printf("matmul dispatch fail\n"); return 1;
    }
    std::vector<float> got(M * N);
    stream.MemcpyDtoh(cr.value, 0, got.data(), got.size() * 4, true);
    std::vector<double> ad(av.begin(), av.end()), bd(bv.begin(), bv.end()), cd(M * N);
    // float64 reference in logical (untransposed) layout
    for (int64_t i = 0; i < M; ++i) for (int64_t j = 0; j < N; ++j) {
      double acc = 0;
      for (int64_t kk = 0; kk < K; ++kk) acc += ad[i * K + kk] * bd[kk * N + j];
      cd[i * N + j] = acc;
    }
    double mx = 1; for (double v : cd) mx = std::max(mx, std::abs(v));
    double tol = 1e-4 * std::sqrt((double)K) * mx;
    for (size_t i = 0; i < got.size(); ++i)
      if (std::abs(got[i] - cd[i]) > tol) { printf("FAIL iter %d idx %zu got %f want %f\n", iter, i, got[i], cd[i]); ++failures; break; }
    alloc.Release(a, stream.last_submitted_token());
    alloc.Release(b, stream.last_submitted_token());
    alloc.Release(cr.value, stream.last_submitted_token());
  }
  printf(failures ? "FUZZ FAILED (%d)\n" : "FUZZ PASS (60 shapes)\n", failures);
  return failures ? 1 : 0;
#else
  printf("SKIP built without Vulkan\n");
  return 0;
#endif
}
