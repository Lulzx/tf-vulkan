// Vulkan compute probe (spec §4 F3 / task 002): feature report + tier.
// Usage: tfvk_probe [--json] [--pipeline-cache PATH]
#include <cstdio>
#include <cstring>

#include "../runtime/device.h"

int main(int argc, char** argv) {
  bool json = false;
  std::string cache;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) json = true;
    if (std::strcmp(argv[i], "--pipeline-cache") == 0 && i + 1 < argc) cache = argv[++i];
  }
#ifndef TFVK_HAS_VULKAN
  printf(json ? "{\"tier\":\"C\",\"reason\":\"built without Vulkan\"}\n"
              : "Tier C: built without Vulkan headers/loader\n");
  return 2;
#else
  tfvk::VulkanDevice dev;
  auto s = dev.Init(cache);
  if (!s.ok()) {
    printf(json ? "{\"tier\":\"C\",\"reason\":\"%s\"}\n" : "Tier C: %s\n", s.message.c_str());
    return 2;
  }
  const auto& i = dev.info();
  const char* tier = i.tier == tfvk::Tier::kA ? "A" : i.tier == tfvk::Tier::kB ? "B" : "C";
  uint64_t heap = i.heap_sizes.empty() ? 0 : i.heap_sizes[0];
  uint64_t budget = tfvk::ComputeBudgetBytes(heap, i.memory_budget_ext, i.budget_heap_size);
  if (json) {
    printf("{\"device\":\"%s\",\"vendor_id\":%u,\"device_id\":%u,\"driver_version\":%u,"
           "\"api_version\":\"%u.%u.%u\",\"compute_queue\":%s,\"timeline_semaphore\":%s,"
           "\"scalar_block_layout\":%s,\"synchronization2\":%s,\"shader_float16\":%s,"
           "\"push_descriptor\":%s,\"max_push_constants\":%u,\"memory_budget_ext\":%s,"
           "\"heap_bytes\":%llu,\"budget_bytes\":%llu,\"tier\":\"%s\",\"tier_reason\":\"%s\"}\n",
           i.name.c_str(), i.vendor_id, i.device_id, i.driver_version,
           VK_VERSION_MAJOR(i.api_version), VK_VERSION_MINOR(i.api_version),
           VK_VERSION_PATCH(i.api_version), i.compute_queue ? "true" : "false",
           i.timeline_semaphore ? "true" : "false", i.scalar_block_layout ? "true" : "false",
           i.synchronization2 ? "true" : "false", i.shader_float16 ? "true" : "false",
           i.push_descriptor ? "true" : "false", i.max_push_constants,
           i.memory_budget_ext ? "true" : "false",
           (unsigned long long)heap, (unsigned long long)budget, tier, i.tier_reason.c_str());
  } else {
    printf("device: %s (vendor %u device %u driver %u)\n", i.name.c_str(), i.vendor_id, i.device_id, i.driver_version);
    printf("Vulkan API: %u.%u.%u  compute_queue=%d\n", VK_VERSION_MAJOR(i.api_version),
           VK_VERSION_MINOR(i.api_version), VK_VERSION_PATCH(i.api_version), (int)i.compute_queue);
    printf("timelineSemaphore=%d scalarBlockLayout=%d synchronization2=%d shaderFloat16=%d pushDescriptor=%d\n",
           (int)i.timeline_semaphore, (int)i.scalar_block_layout, (int)i.synchronization2,
           (int)i.shader_float16, (int)i.push_descriptor);
    printf("memory types:\n");
    for (auto& m : i.memory_types)
      printf("  [%u] heap=%u device_local=%d host_visible=%d coherent=%d cached=%d\n", m.index,
             m.heap_index, (int)m.device_local, (int)m.host_visible, (int)m.host_coherent, (int)m.host_cached);
    printf("budget: heap=%llu bytes, advertised=%llu bytes (memory_budget_ext=%d)\n",
           (unsigned long long)heap, (unsigned long long)budget, (int)i.memory_budget_ext);
    printf("Tier %s: %s\n", tier, i.tier_reason.c_str());
  }
  dev.SavePipelineCache();
  return i.tier == tfvk::Tier::kC ? 2 : 0;
#endif
}
