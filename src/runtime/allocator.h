#pragma once
#include <cstdint>
#include <map>
#include <vector>

#include "device.h"
#include "status.h"

namespace tfvk {

// Pooled VkBuffer allocator (spec §6.2). Buffers are recycled only after the
// completion token (fence/timeline value) of their last use has passed.
struct Allocation {
  uint64_t id = 0;
#ifdef TFVK_HAS_VULKAN
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  void* mapped = nullptr;
#endif
  uint64_t size = 0;       // requested
  uint64_t capacity = 0;   // rounded allocation size
  uint64_t last_use_token = 0;
  bool host_coherent = true;
};

struct AllocatorStats {
  uint64_t allocations = 0, reuses = 0, frees = 0, bytes_in_use = 0, peak_bytes = 0;
  uint64_t h2d_bytes = 0, d2h_bytes = 0, h2d_count = 0, d2h_count = 0;
};

class Allocator {
 public:
#ifdef TFVK_HAS_VULKAN
  Allocator(VulkanDevice* dev, uint64_t budget_bytes) : dev_(dev), budget_(budget_bytes) {}
  Result<Allocation*> Allocate(uint64_t size, uint64_t completed_token);
  void Release(Allocation* a, uint64_t last_use_token);
  // Destroy pooled buffers whose last use completed at or before token.
  void Collect(uint64_t completed_token);
  ~Allocator();
#endif
  uint64_t budget() const { return budget_; }
  const AllocatorStats& stats() const { return stats_; }
  void NoteH2D(uint64_t n) { stats_.h2d_bytes += n; ++stats_.h2d_count; }
  void NoteD2H(uint64_t n) { stats_.d2h_bytes += n; ++stats_.d2h_count; }

 private:
  uint64_t RoundSize(uint64_t s) const { return (s + 4095) & ~uint64_t(4095); }
#ifdef TFVK_HAS_VULKAN
  VulkanDevice* dev_ = nullptr;
  std::multimap<uint64_t, Allocation*> free_;  // by capacity
  std::vector<Allocation*> live_;
#endif
  uint64_t budget_ = 0;
  AllocatorStats stats_;
  uint64_t next_id_ = 1;
};

}  // namespace tfvk
