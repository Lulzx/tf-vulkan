#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "allocator.h"
#include "device.h"
#include "status.h"

namespace tfvk {

enum class FlushReason { kSync, kRecordEvent, kDtoh, kDispatchThreshold, kWorkThreshold, kExplicit };

// Deferred-submission stream (spec §6.3). Ops record into one open command
// buffer; mandatory flush triggers close+submit it with a fence token.
struct StreamConfig {
  uint32_t max_dispatches = 64;          // flush threshold (spec: 32-64)
  uint64_t max_workgroups = 1u << 20;    // watchdog bound per submission (§12)
};

struct Event {
  uint64_t token = 0;  // fence sequence value that must complete
};

struct StreamStats {
  uint64_t submissions = 0, dispatches = 0;
  uint64_t flushes_sync = 0, flushes_event = 0, flushes_dtoh = 0;
  uint64_t flushes_dispatch_threshold = 0, flushes_work_threshold = 0, flushes_explicit = 0;
};

#ifdef TFVK_HAS_VULKAN
class Stream {
 public:
  Stream(VulkanDevice* dev, Allocator* alloc, StreamConfig cfg = {});
  ~Stream();
  Status PrepareBatch() { return EnsureOpen(); }
  // Record a dispatch; record_fn writes commands into the open cmd buffer.
  Status Record(std::function<void(VkCommandBuffer)> fn, uint64_t est_workgroups = 1);
  Status Flush(FlushReason reason);
  Status BlockHostUntilDone();                 // SP block_host_until_done / synchronize_all_activity
  Result<Event> RecordEvent();                 // SP record_event: flush + token
  bool EventReady(const Event& e);
  // UMA hazard rules (spec §6.2): dtoh flushes+waits if buffer has pending writes;
  // htod is immediate only for idle buffers.
  Status MemcpyDtoh(Allocation* src, uint64_t offset, void* dst, uint64_t size, bool src_pending);
  Status MemcpyHtod(const void* src, Allocation* dst, uint64_t offset, uint64_t size, bool dst_pending);
  uint64_t completed_token();
  uint64_t last_submitted_token() const { return submitted_; }
  const StreamStats& stats() const { return stats_; }
  VkCommandBuffer cmd() const { return cmd_; }
  // Per-batch ring descriptor pool (spec §6.3), reset after its fence signals.
  VkDescriptorPool descriptor_pool() const { return pool_; }
  VulkanDevice* device() const { return dev_; }

 private:
  Status EnsureOpen();
  VulkanDevice* dev_;
  Allocator* alloc_;
  StreamConfig cfg_;
  VkCommandPool pool_cmd_ = VK_NULL_HANDLE;
  VkCommandBuffer cmd_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_ = VK_NULL_HANDLE;
  struct InFlight { VkFence fence; uint64_t token; VkDescriptorPool pool; VkCommandBuffer cmd; };
  std::vector<InFlight> in_flight_;
  uint64_t submitted_ = 0, completed_ = 0;
  uint32_t batch_dispatches_ = 0;
  uint64_t batch_workgroups_ = 0;
  bool open_ = false;
  StreamStats stats_;
};
#endif

}  // namespace tfvk
