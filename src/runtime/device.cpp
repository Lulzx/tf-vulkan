#include "device.h"

#include <cstdio>
#include <cstring>

namespace tfvk {

uint64_t ComputeBudgetBytes(uint64_t heap_size, bool budget_ext, uint64_t ext_budget) {
  if (budget_ext && ext_budget > 0) return ext_budget;
  return heap_size * 30 / 100;
}

#ifdef TFVK_HAS_VULKAN
VulkanDevice::~VulkanDevice() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    if (pipeline_cache_ != VK_NULL_HANDLE) vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) vkDestroyInstance(instance_, nullptr);
}

static Tier Classify(const DeviceInfo& i, std::string* why) {
  if (!i.compute_queue) { *why = "no compute queue"; return Tier::kC; }
  bool baseline = i.api_version >= VK_API_VERSION_1_3 && i.timeline_semaphore &&
                  i.scalar_block_layout && i.synchronization2;
  if (baseline) { *why = "Vulkan 1.3 IREE baseline met"; return Tier::kA; }
  *why = "compute present, IREE Vulkan 1.3 baseline missing (native path only)";
  return Tier::kB;
}

Status VulkanDevice::Init(const std::string& pipeline_cache_path) {
  pipeline_cache_path_ = pipeline_cache_path;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "tf-vulkan";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
#ifdef __APPLE__
  // MoltenVK portability (desktop dev harness only; Android needs none of this).
  ici.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  const char* exts[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
                        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
  ici.enabledExtensionCount = 2;
  ici.ppEnabledExtensionNames = exts;
#endif
  if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
    // Retry at Vulkan 1.1 (Tier B drivers).
    app.apiVersion = VK_API_VERSION_1_1;
    if (vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS)
      return Status::Error(Code::kUnavailable, "vkCreateInstance failed");
  }
  uint32_t count = 0;
  vkEnumeratePhysicalDevices(instance_, &count, nullptr);
  if (count == 0) return Status::Error(Code::kUnavailable, "no Vulkan physical devices");
  std::vector<VkPhysicalDevice> devs(count);
  vkEnumeratePhysicalDevices(instance_, &count, devs.data());

  for (VkPhysicalDevice pd : devs) {
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qcount, qprops.data());
    for (uint32_t q = 0; q < qcount; ++q) {
      if (!(qprops[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
      physical_ = pd;
      queue_family_ = q;
      info_.compute_queue = true;
      break;
    }
    if (physical_ != VK_NULL_HANDLE) break;
  }
  if (physical_ == VK_NULL_HANDLE)
    return Status::Error(Code::kUnavailable, "no compute queue family");

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_, &props);
  info_.name = props.deviceName;
  info_.vendor_id = props.vendorID;
  info_.device_id = props.deviceID;
  info_.driver_version = props.driverVersion;
  info_.api_version = props.apiVersion;
  info_.max_push_constants = props.limits.maxPushConstantsSize;
  info_.max_workgroup_x = props.limits.maxComputeWorkGroupSize[0];
  info_.pipeline_cache_uuid.assign(props.pipelineCacheUUID,
                                   props.pipelineCacheUUID + VK_UUID_SIZE);

  // Feature probe (never inferred from marketing name, spec §5).
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  f12.pNext = &f13;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  f11.pNext = &f12;
  VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  feats.pNext = &f11;
  vkGetPhysicalDeviceFeatures2(physical_, &feats);
  info_.timeline_semaphore = f12.timelineSemaphore == VK_TRUE;
  info_.scalar_block_layout = f12.scalarBlockLayout == VK_TRUE;
  info_.synchronization2 = f13.synchronization2 == VK_TRUE;
  {
    // Real FP16 gate (spec §6.4): shaderFloat16 + 16-bit storage access.
    VkPhysicalDeviceShaderFloat16Int8Features f16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16.pNext = &f16;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &s16;
    vkGetPhysicalDeviceFeatures2(physical_, &f2);
    info_.shader_float16 = f16.shaderFloat16 == VK_TRUE &&
                           s16.storageBuffer16BitAccess == VK_TRUE;
  }

  uint32_t ecount = 0;
  vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ecount, nullptr);
  std::vector<VkExtensionProperties> extprops(ecount);
  vkEnumerateDeviceExtensionProperties(physical_, nullptr, &ecount, extprops.data());
  std::vector<const char*> enabled;
  for (auto& e : extprops) {
    std::string n = e.extensionName;
    if (n == VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) info_.memory_budget_ext = true;
    if (n == VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) {
      info_.push_descriptor = true;
      enabled.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    }
#ifdef __APPLE__
    if (n == "VK_KHR_portability_subset")
      enabled.push_back("VK_KHR_portability_subset");
#endif
  }
  if (info_.synchronization2) enabled.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(physical_, &mem);
  for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) info_.heap_sizes.push_back(mem.memoryHeaps[i].size);
  for (uint32_t i = 0; i < mem.memoryTypeCount; ++i) {
    auto f = mem.memoryTypes[i].propertyFlags;
    info_.memory_types.push_back({i, !!(f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
                                  !!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT),
                                  !!(f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                                  !!(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT),
                                  mem.memoryTypes[i].heapIndex});
  }
  // VK_EXT_memory_budget heap budget (spec §6.1).
  if (info_.memory_budget_ext) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    mp2.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(physical_, &mp2);
    info_.budget_heap_size = budget.heapBudget[0];
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = queue_family_;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkPhysicalDeviceSynchronization2FeaturesKHR sync2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
  sync2.synchronization2 = VK_TRUE;
  // Enable FP16 only when the probe passed (opt-in K1.5, §6.4).
  VkPhysicalDeviceShaderFloat16Int8Features f16feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
  VkPhysicalDevice16BitStorageFeatures s16feat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
  if (info_.shader_float16) {
    f16feat.shaderFloat16 = VK_TRUE;
    s16feat.storageBuffer16BitAccess = VK_TRUE;
    s16feat.pNext = &f16feat;
    sync2.pNext = &s16feat;
  }
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = static_cast<uint32_t>(enabled.size());
  dci.ppEnabledExtensionNames = enabled.data();
  if (info_.synchronization2) dci.pNext = &sync2;
  if (vkCreateDevice(physical_, &dci, nullptr, &device_) != VK_SUCCESS)
    return Status::Error(Code::kUnavailable, "vkCreateDevice failed");
  vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

  // Persistent pipeline cache keyed by driver UUID (spec §12): discard on mismatch.
  std::vector<char> cache_data;
  if (!pipeline_cache_path_.empty()) {
    FILE* f = fopen(pipeline_cache_path_.c_str(), "rb");
    if (f) {
      fseek(f, 0, SEEK_END);
      long sz = ftell(f);
      fseek(f, 0, SEEK_SET);
      if (sz > 16) {
        cache_data.resize(sz);
        if (fread(cache_data.data(), 1, sz, f) == static_cast<size_t>(sz)) {
          // Header: uint32 version, uint32 vendor, uint32 device, UUID (16B)
          uint32_t vendor = 0, device = 0;
          memcpy(&vendor, cache_data.data() + 8, 4);
          memcpy(&device, cache_data.data() + 12, 4);
          bool uuid_ok = sz >= 32 && memcmp(cache_data.data() + 16, props.pipelineCacheUUID, 16) == 0;
          if (vendor != props.vendorID || device != props.deviceID || !uuid_ok) cache_data.clear();
        } else cache_data.clear();
      }
      fclose(f);
    }
  }
  VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
  pcci.initialDataSize = cache_data.size();
  pcci.pInitialData = cache_data.empty() ? nullptr : cache_data.data();
  vkCreatePipelineCache(device_, &pcci, nullptr, &pipeline_cache_);

  info_.tier = Classify(info_, &info_.tier_reason);
  return Status::Ok();
}

Status VulkanDevice::SavePipelineCache() {
  if (pipeline_cache_path_.empty() || pipeline_cache_ == VK_NULL_HANDLE) return Status::Ok();
  size_t sz = 0;
  if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, nullptr) != VK_SUCCESS || sz == 0)
    return Status::Ok();
  std::vector<char> data(sz);
  if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, data.data()) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkGetPipelineCacheData failed");
  FILE* f = fopen(pipeline_cache_path_.c_str(), "wb");
  if (!f) return Status::Error(Code::kInternal, "cannot write pipeline cache");
  fwrite(data.data(), 1, sz, f);
  fclose(f);
  return Status::Ok();
}

Result<uint32_t> VulkanDevice::FindMemoryType(uint32_t type_bits, bool prefer_uma) const {
  int32_t fallback = -1;
  for (uint32_t i = 0; i < info_.memory_types.size(); ++i) {
    if (!(type_bits & (1u << i))) continue;
    auto& m = info_.memory_types[i];
    if (prefer_uma && m.device_local && m.host_visible) return {Status::Ok(), i};
    if (m.host_visible && fallback < 0) fallback = static_cast<int32_t>(i);
    if (!prefer_uma && fallback < 0) fallback = static_cast<int32_t>(i);
  }
  if (fallback >= 0) return {Status::Ok(), static_cast<uint32_t>(fallback)};
  return {Status::Error(Code::kResourceExhausted, "no suitable memory type"), 0};
}
#endif

}  // namespace tfvk
