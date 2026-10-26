#include "allocator.h"

#include <algorithm>
#include <cstring>

namespace tfvk {

#ifdef TFVK_HAS_VULKAN
Allocator::~Allocator() {
  Collect(UINT64_MAX);
  for (auto* a : live_) {
    vkDestroyBuffer(dev_->device(), a->buffer, nullptr);
    vkFreeMemory(dev_->device(), a->memory, nullptr);
    delete a;
  }
}

Result<Allocation*> Allocator::Allocate(uint64_t size, uint64_t completed_token) {
  uint64_t cap = RoundSize(size == 0 ? 4096 : size);
  // Reuse only buffers whose last GPU use has completed (spec §6.2).
  for (auto it = free_.lower_bound(cap); it != free_.end(); ++it) {
    if (it->first <= cap * 2 && it->second->last_use_token <= completed_token) {
      Allocation* a = it->second;
      free_.erase(it);
      a->size = size;
      ++stats_.reuses;
      stats_.bytes_in_use += a->capacity;
      stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.bytes_in_use);
      return {Status::Ok(), a};
    }
  }
  if (stats_.bytes_in_use + cap > budget_)
    return {Status::Error(Code::kResourceExhausted, "tf-vulkan GPU budget exceeded"), nullptr};

  auto* a = new Allocation();
  a->id = next_id_++;
  a->size = size;
  a->capacity = cap;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = cap;
  bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
              VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(dev_->device(), &bci, nullptr, &a->buffer) != VK_SUCCESS) {
    delete a;
    return {Status::Error(Code::kInternal, "vkCreateBuffer failed"), nullptr};
  }
  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(dev_->device(), a->buffer, &req);
  auto mt = dev_->FindMemoryType(req.memoryTypeBits, /*prefer_uma=*/true);
  if (!mt.ok()) {
    vkDestroyBuffer(dev_->device(), a->buffer, nullptr);
    delete a;
    return {mt.status, nullptr};
  }
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = mt.value;
  if (vkAllocateMemory(dev_->device(), &mai, nullptr, &a->memory) != VK_SUCCESS) {
    vkDestroyBuffer(dev_->device(), a->buffer, nullptr);
    delete a;
    return {Status::Error(Code::kResourceExhausted, "vkAllocateMemory failed (OOM)"), nullptr};
  }
  vkBindBufferMemory(dev_->device(), a->buffer, a->memory, 0);
  for (auto& m : dev_->info().memory_types) {
    if (m.index == mt.value) {
      a->host_coherent = m.host_coherent;
      if (m.host_visible)
        vkMapMemory(dev_->device(), a->memory, 0, VK_WHOLE_SIZE, 0, &a->mapped);
    }
  }
  live_.push_back(a);
  ++stats_.allocations;
  stats_.bytes_in_use += cap;
  stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.bytes_in_use);
  return {Status::Ok(), a};
}

void Allocator::Release(Allocation* a, uint64_t last_use_token) {
  if (!a) return;
  a->last_use_token = std::max(a->last_use_token, last_use_token);
  stats_.bytes_in_use -= std::min(stats_.bytes_in_use, a->capacity);
  ++stats_.frees;
  free_.emplace(a->capacity, a);
}

void Allocator::Collect(uint64_t completed_token) {
  for (auto it = free_.begin(); it != free_.end();) {
    if (it->second->last_use_token <= completed_token && free_.size() > 16) {
      Allocation* a = it->second;
      it = free_.erase(it);
      vkDestroyBuffer(dev_->device(), a->buffer, nullptr);
      vkFreeMemory(dev_->device(), a->memory, nullptr);
      live_.erase(std::remove(live_.begin(), live_.end(), a), live_.end());
      delete a;
    } else ++it;
  }
}
#endif

}  // namespace tfvk
