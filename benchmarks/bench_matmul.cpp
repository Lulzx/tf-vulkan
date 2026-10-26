// MatMul benchmark (spec §11): warmup + median/p10/p90, whole-call latency
// including transfer and sync; GPU dispatch confirmed via fence completion.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/stream.h"

static double Percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p * v.size()))];
}

int main(int argc, char** argv) {
#ifndef TFVK_HAS_VULKAN
  printf("Tier C: built without Vulkan\n");
  return 2;
#else
  tfvk::VulkanDevice dev;
  if (!dev.Init("").ok()) { printf("Tier C: no Vulkan device\n"); return 2; }
  uint64_t heap = dev.info().heap_sizes.empty() ? 0 : dev.info().heap_sizes[0];
  tfvk::Allocator alloc(&dev, tfvk::ComputeBudgetBytes(heap, dev.info().memory_budget_ext, dev.info().budget_heap_size));
  tfvk::Stream stream(&dev, &alloc);
  tfvk::VulkanKernels k(&dev, &stream);
  if (!k.Init().ok()) { printf("kernels unavailable\n"); return 2; }
  printf("device=%s tier=%d (desktop harness, not a mobile claim, spec §11)\n", dev.info().name.c_str(), (int)dev.info().tier);
  std::vector<int> sizes = {128, 512, 1024};
  if (argc > 1) sizes = {atoi(argv[1])};
  std::mt19937 rng(7);
  std::normal_distribution<float> nd;
  for (int S : sizes) {
    std::vector<float> av((size_t)S * S), bv((size_t)S * S);
    for (auto& x : av) x = nd(rng);
    for (auto& x : bv) x = nd(rng);
    auto A = alloc.Allocate(av.size() * 4, 0); auto B = alloc.Allocate(bv.size() * 4, 0); auto C = alloc.Allocate(av.size() * 4, 0);
    if (!A.ok() || !B.ok() || !C.ok()) { printf("%d: alloc failed\n", S); continue; }
    stream.MemcpyHtod(av.data(), A.value, 0, av.size() * 4, false);
    stream.MemcpyHtod(bv.data(), B.value, 0, bv.size() * 4, false);
    auto run = [&]() {
      k.MatMul(A.value, B.value, C.value, S, S, S);
      stream.BlockHostUntilDone();
    };
    for (int i = 0; i < 3; ++i) run();  // warm pipeline
    std::vector<double> gpu_ms, cpu_ms;
    for (int i = 0; i < 15; ++i) {
      auto t0 = std::chrono::steady_clock::now(); run();
      gpu_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<float> cref((size_t)S * S);
    for (int i = 0; i < 3; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      tfvk::CpuMatMul(av.data(), bv.data(), cref.data(), S, S, S);
      cpu_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<float> got((size_t)S);
    stream.MemcpyDtoh(C.value, 0, got.data(), S * 4, false);
    double err = std::abs(got[0] - cref[0]);
    printf("MatMul %dx%dx%d  GPU median %.3f ms (p10 %.3f p90 %.3f)  CPU median %.3f ms  speedup %.2fx  sample_err %.2e\n",
           S, S, S, Percentile(gpu_ms, .5), Percentile(gpu_ms, .1), Percentile(gpu_ms, .9),
           Percentile(cpu_ms, .5), Percentile(cpu_ms, .5) / Percentile(gpu_ms, .5), err);
    alloc.Release(A.value, 0); alloc.Release(B.value, 0); alloc.Release(C.value, 0);
  }
  return 0;
#endif
}
