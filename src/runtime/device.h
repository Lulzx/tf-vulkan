#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "status.h"

#ifdef TFVK_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace tfvk {

// Capability tier (spec §5): A = IREE Vulkan 1.3 baseline, B = native-only, C = disabled.
enum class Tier { kA, kB, kC };

struct MemoryTypeInfo {
  uint32_t index = 0;
  bool device_local = false, host_visible = false, host_coherent = false, host_cached = false;
  uint32_t heap_index = 0;
};

struct DeviceInfo {
  std::string name;
  uint32_t vendor_id = 0, device_id = 0, driver_version = 0, api_version = 0;
  bool compute_queue = false;
  bool timeline_semaphore = false, scalar_block_layout = false;
  bool synchronization2 = false, shader_float16 = false, push_descriptor = false;
  uint32_t max_push_constants = 0, max_workgroup_x = 0;
  std::vector<uint8_t> pipeline_cache_uuid;
  std::vector<MemoryTypeInfo> memory_types;
  std::vector<uint64_t> heap_sizes;
  bool memory_budget_ext = false;
  uint64_t budget_heap_size = 0;  // from VK_EXT_memory_budget when present
  Tier tier = Tier::kC;
  std::string tier_reason;
};

// Conservative advertised budget (spec §6.1): VK_EXT_memory_budget when
// available, else 25-40% of the heap (we use 30%).
uint64_t ComputeBudgetBytes(uint64_t heap_size, bool budget_ext, uint64_t ext_budget);

#ifdef TFVK_HAS_VULKAN
class VulkanDevice {
 public:
  ~VulkanDevice();
  Status Init(const std::string& pipeline_cache_path = "");
  bool valid() const { return device_ != VK_NULL_HANDLE; }
  const DeviceInfo& info() const { return info_; }
  VkInstance instance() const { return instance_; }
  VkPhysicalDevice physical() const { return physical_; }
  VkDevice device() const { return device_; }
  VkQueue queue() const { return queue_; }
  uint32_t queue_family() const { return queue_family_; }
  // Picks a memory type; prefers DEVICE_LOCAL|HOST_VISIBLE on UMA (spec §6.2).
  Result<uint32_t> FindMemoryType(uint32_t type_bits, bool prefer_uma) const;
  Status SavePipelineCache();
  VkPipelineCache pipeline_cache() const { return pipeline_cache_; }
  bool device_lost() const { return device_lost_; }
  void MarkDeviceLost() { device_lost_ = true; }
 private:
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physical_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
  std::string pipeline_cache_path_;
  DeviceInfo info_;
  bool device_lost_ = false;
};
#endif

}  // namespace tfvk
