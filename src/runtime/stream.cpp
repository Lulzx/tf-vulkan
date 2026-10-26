#include "stream.h"

#include "tensor.h"

#include <cstring>

namespace tfvk {

#ifdef TFVK_HAS_VULKAN
static VkDescriptorPool MakePool(VkDevice d, uint32_t max_sets) {
  VkDescriptorPoolSize sz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, max_sets * 4};
  VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  ci.maxSets = max_sets;
  ci.poolSizeCount = 1;
  ci.pPoolSizes = &sz;
  VkDescriptorPool p = VK_NULL_HANDLE;
  vkCreateDescriptorPool(d, &ci, nullptr, &p);
  return p;
}

Stream::Stream(VulkanDevice* dev, Allocator* alloc, StreamConfig cfg)
    : dev_(dev), alloc_(alloc), cfg_(cfg) {
  VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  ci.queueFamilyIndex = dev_->queue_family();
  vkCreateCommandPool(dev_->device(), &ci, nullptr, &pool_cmd_);
  // Command buffers are allocated per batch in EnsureOpen and freed when
  // their fence signals: re-recording a still-in-flight buffer is invalid.
  pool_ = MakePool(dev_->device(), cfg_.max_dispatches + 1);
}

Stream::~Stream() {
  if (dev_->valid()) {
    vkDeviceWaitIdle(dev_->device());
    for (auto& f : in_flight_) {
      vkDestroyFence(dev_->device(), f.fence, nullptr);
      if (f.cmd != VK_NULL_HANDLE) vkFreeCommandBuffers(dev_->device(), pool_cmd_, 1, &f.cmd);
      if (f.pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev_->device(), f.pool, nullptr);
    }
    if (cmd_ != VK_NULL_HANDLE) vkFreeCommandBuffers(dev_->device(), pool_cmd_, 1, &cmd_);
    if (pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev_->device(), pool_, nullptr);
    if (pool_cmd_ != VK_NULL_HANDLE) vkDestroyCommandPool(dev_->device(), pool_cmd_, nullptr);
  }
}

Status Stream::EnsureOpen() {
  if (open_) return Status::Ok();
  // Retire completed in-flight batches; reset their descriptor pools (spec §6.3).
  for (size_t i = 0; i < in_flight_.size();) {
    if (vkGetFenceStatus(dev_->device(), in_flight_[i].fence) == VK_SUCCESS) {
      completed_ = std::max(completed_, in_flight_[i].token);
      vkDestroyFence(dev_->device(), in_flight_[i].fence, nullptr);
      if (in_flight_[i].cmd != VK_NULL_HANDLE)
        vkFreeCommandBuffers(dev_->device(), pool_cmd_, 1, &in_flight_[i].cmd);
      if (in_flight_[i].pool != VK_NULL_HANDLE) {
        // Reuse the retired pool as the next batch pool.
        vkResetDescriptorPool(dev_->device(), in_flight_[i].pool, 0);
        if (pool_ == VK_NULL_HANDLE) pool_ = in_flight_[i].pool;
        else vkDestroyDescriptorPool(dev_->device(), in_flight_[i].pool, nullptr);
      }
      in_flight_.erase(in_flight_.begin() + static_cast<long>(i));
    } else ++i;
  }
  if (pool_ == VK_NULL_HANDLE) pool_ = MakePool(dev_->device(), cfg_.max_dispatches + 1);
  // Fresh command buffer per batch (never re-record an in-flight one).
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool_cmd_;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  cmd_ = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(dev_->device(), &ai, &cmd_) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkAllocateCommandBuffers failed");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(cmd_, &bi) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkBeginCommandBuffer failed");
  open_ = true;
  batch_dispatches_ = 0;
  batch_workgroups_ = 0;
  return Status::Ok();
}

Status Stream::Record(std::function<void(VkCommandBuffer)> fn, uint64_t est_workgroups) {
  auto s = EnsureOpen();
  if (!s.ok()) return s;
  fn(cmd_);
  ++batch_dispatches_;
  ++stats_.dispatches;
  batch_workgroups_ += est_workgroups;
  // Bounded batch: dispatch-count and watchdog work thresholds (§6.3/§12).
  if (batch_dispatches_ >= cfg_.max_dispatches) return Flush(FlushReason::kDispatchThreshold);
  if (batch_workgroups_ >= cfg_.max_workgroups) return Flush(FlushReason::kWorkThreshold);
  return Status::Ok();
}

Status Stream::Flush(FlushReason reason) {
  if (!open_) return Status::Ok();
  vkEndCommandBuffer(cmd_);
  VkFence fence = VK_NULL_HANDLE;
  VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(dev_->device(), &fi, nullptr, &fence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd_;
  VkResult r = vkQueueSubmit(dev_->queue(), 1, &si, fence);
  if (r == VK_ERROR_DEVICE_LOST) {
    dev_->MarkDeviceLost();
    return Status::Error(Code::kDeviceLost, "VK_ERROR_DEVICE_LOST on submit");
  }
  if (r != VK_SUCCESS) return Status::Error(Code::kInternal, "vkQueueSubmit failed");
  open_ = false;
  ++submitted_;
  ++stats_.submissions;
  switch (reason) {
    case FlushReason::kSync: ++stats_.flushes_sync; break;
    case FlushReason::kRecordEvent: ++stats_.flushes_event; break;
    case FlushReason::kDtoh: ++stats_.flushes_dtoh; break;
    case FlushReason::kDispatchThreshold: ++stats_.flushes_dispatch_threshold; break;
    case FlushReason::kWorkThreshold: ++stats_.flushes_work_threshold; break;
    case FlushReason::kExplicit: ++stats_.flushes_explicit; break;
  }
  in_flight_.push_back({fence, submitted_, pool_, cmd_});
  pool_ = VK_NULL_HANDLE;  // owned by in-flight batch until its fence signals
  cmd_ = VK_NULL_HANDLE;   // owned by in-flight batch until its fence signals
  alloc_->Collect(completed_token());
  return Status::Ok();
}

uint64_t Stream::completed_token() {
  for (auto& f : in_flight_)
    if (vkGetFenceStatus(dev_->device(), f.fence) == VK_SUCCESS)
      completed_ = std::max(completed_, f.token);
  return completed_;
}

Status Stream::BlockHostUntilDone() {
  auto s = Flush(FlushReason::kSync);
  if (!s.ok()) return s;
  for (auto& f : in_flight_) {
    if (vkWaitForFences(dev_->device(), 1, &f.fence, VK_TRUE, UINT64_MAX) == VK_ERROR_DEVICE_LOST) {
      dev_->MarkDeviceLost();
      return Status::Error(Code::kDeviceLost, "VK_ERROR_DEVICE_LOST on wait");
    }
    completed_ = std::max(completed_, f.token);
  }
  return Status::Ok();
}

Result<Event> Stream::RecordEvent() {
  auto s = Flush(FlushReason::kRecordEvent);
  if (!s.ok()) return {s, {}};
  return {Status::Ok(), Event{submitted_}};
}

bool Stream::EventReady(const Event& e) { return completed_token() >= e.token; }

Status Stream::MemcpyDtoh(Allocation* src, uint64_t offset, void* dst, uint64_t size, bool src_pending) {
  auto chk = CheckRange(src->capacity, offset, size);
  if (!chk.ok()) return chk;
  if (src_pending) {  // never memcpy from a buffer with unsubmitted/in-flight writes (§6.2)
    auto s = BlockHostUntilDone();
    if (!s.ok()) return s;
  }
  if (src->mapped) {
    if (!src->host_coherent) {
      VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
      range.memory = src->memory;
      range.offset = offset;
      range.size = size;
      vkInvalidateMappedMemoryRanges(dev_->device(), 1, &range);
    }
    memcpy(dst, static_cast<char*>(src->mapped) + offset, size);
  } else {
    return Status::Error(Code::kUnsupported, "staging dtoh path not available for non-mapped buffer");
  }
  alloc_->NoteD2H(size);
  return Status::Ok();
}

Status Stream::MemcpyHtod(const void* src, Allocation* dst, uint64_t offset, uint64_t size, bool dst_pending) {
  auto chk = CheckRange(dst->capacity, offset, size);
  if (!chk.ok()) return chk;
  if (dst_pending) {  // in-place update of a busy buffer must synchronize first (§6.2)
    auto s = BlockHostUntilDone();
    if (!s.ok()) return s;
  }
  if (!dst->mapped) return Status::Error(Code::kUnsupported, "staging htod path not available");
  memcpy(static_cast<char*>(dst->mapped) + offset, src, size);
  if (!dst->host_coherent) {
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = dst->memory;
    range.offset = offset;
    range.size = size;
    vkFlushMappedMemoryRanges(dev_->device(), 1, &range);
  }
  alloc_->NoteH2D(size);
  return Status::Ok();
}
#endif

}  // namespace tfvk
