// TensorFlow PluggableDevice plugin for TF Vulkan (spec §5, §18 PR#2/PR#3).
// Written against the pinned TensorFlow C ABI (validated with TF 2.21 wheel
// headers; see docs/api-compatibility.md). Device: type from
// TF_VULKAN_DEVICE_TYPE (default "GPU", D1 provisional), subtype "VULKAN".
// Tensor data lives in host-mapped Vulkan buffers (UMA path, §6.2), so
// TF_TensorData pointers resolve to our Allocation registry.
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "tensorflow/c/experimental/grappler/grappler.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_buffer.h"
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/kernels_experimental.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#include "../kernels/kernels.h"
#include "../runtime/stream.h"

namespace {
struct State {
  tfvk::VulkanDevice device;
  std::unique_ptr<tfvk::Allocator> allocator;
  std::unique_ptr<tfvk::Stream> stream;
  std::unique_ptr<tfvk::VulkanKernels> kernels;
  std::unordered_map<void*, tfvk::Allocation*> registry;  // mapped ptr -> alloc
  std::mutex mu;
  bool ready = false;
  std::string hardware_name = "Vulkan GPU";
};
State g;

const char* DeviceType() {
  const char* t = std::getenv("TF_VULKAN_DEVICE_TYPE");
  return (t && *t) ? t : "GPU";  // D1 provisional (§5)
}

bool EnsureInit() {
  std::lock_guard<std::mutex> lk(g.mu);
  if (g.ready) return true;
  if (!g.device.Init().ok()) return false;
  uint64_t heap = g.device.info().heap_sizes.empty() ? 0 : g.device.info().heap_sizes[0];
  g.allocator.reset(new tfvk::Allocator(
      &g.device, tfvk::ComputeBudgetBytes(heap, g.device.info().memory_budget_ext,
                                          g.device.info().budget_heap_size)));
  g.stream.reset(new tfvk::Stream(&g.device, g.allocator.get()));
  g.kernels.reset(new tfvk::VulkanKernels(&g.device, g.stream.get()));
  if (!g.kernels->Init().ok()) return false;
  g.hardware_name = g.device.info().name;
  g.ready = true;
  return true;
}

tfvk::Allocation* Lookup(void* p) {
  std::lock_guard<std::mutex> lk(g.mu);
  auto it = g.registry.find(p);
  return it == g.registry.end() ? nullptr : it->second;
}
void RegisterAlloc(tfvk::Allocation* a) {
  if (a && a->mapped) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.registry[a->mapped] = a;
  }
}
void UnregisterAlloc(tfvk::Allocation* a) {
  if (a && a->mapped) {
    std::lock_guard<std::mutex> lk(g.mu);
    g.registry.erase(a->mapped);
  }
}

// ---------------- StreamExecutor callbacks ----------------
void SEAllocate(const SP_Device*, uint64_t size, int64_t, SP_DeviceMemoryBase* mem) {
  mem->struct_size = SP_DEVICE_MEMORY_BASE_STRUCT_SIZE;
  mem->opaque = nullptr;
  mem->size = size;
  mem->payload = 0;
  if (!EnsureInit()) return;
  auto r = g.allocator->Allocate(size ? size : 1, g.stream->completed_token());
  if (!r.ok()) return;
  RegisterAlloc(r.value);
  mem->opaque = r.value->mapped ? r.value->mapped : (void*)r.value;
  mem->payload = (uint64_t)(uintptr_t)r.value;
}
void SEDeallocate(const SP_Device*, SP_DeviceMemoryBase* memory) {
  if (!memory || !memory->payload) return;
  auto* a = (tfvk::Allocation*)(uintptr_t)memory->payload;
  UnregisterAlloc(a);
  g.allocator->Release(a, g.stream->last_submitted_token());
  memory->opaque = nullptr;
  memory->payload = 0;
}
void* SEHostAlloc(const SP_Device*, uint64_t size) { return std::malloc(size ? size : 1); }
void SEHostDealloc(const SP_Device*, void* mem) { std::free(mem); }
void* SEUnifiedAlloc(const SP_Device*, uint64_t bytes) {
  SP_DeviceMemoryBase m{};
  SEAllocate(nullptr, bytes, 0, &m);
  return m.opaque;
}
void SEUnifiedDealloc(const SP_Device*, void* location) {
  auto* a = Lookup(location);
  if (a) {
    UnregisterAlloc(a);
    g.allocator->Release(a, g.stream->last_submitted_token());
  }
}
TF_Bool SEGetAllocStats(const SP_Device*, SP_AllocatorStats* stats) {
  stats->struct_size = SP_ALLOCATORSTATS_STRUCT_SIZE;
  auto& s = g.allocator->stats();
  stats->num_allocs = (int64_t)s.allocations;
  stats->bytes_in_use = (int64_t)s.bytes_in_use;
  stats->peak_bytes_in_use = (int64_t)s.peak_bytes;
  stats->largest_alloc_size = (int64_t)s.peak_bytes;
  stats->has_bytes_limit = 1;
  stats->bytes_limit = (int64_t)g.allocator->budget();
  return 1;
}
TF_Bool SEDeviceMemUsage(const SP_Device*, int64_t* free_out, int64_t* total_out) {
  *total_out = (int64_t)g.allocator->budget();
  *free_out = *total_out - (int64_t)g.allocator->stats().bytes_in_use;
  return 1;
}
struct StreamToken { int id; };
struct EventObj { uint64_t token = 0; };
struct TimerObj { uint64_t start_ns = 0, stop_ns = 0; };
void SECreateStream(const SP_Device*, SP_Stream* stream, TF_Status*) {
  static int next = 1;
  *stream = (SP_Stream)(void*)new StreamToken{next++};
}
void SEDestroyStream(const SP_Device*, SP_Stream stream) { delete (StreamToken*)(void*)stream; }
void SEStreamDep(const SP_Device*, SP_Stream, SP_Stream, TF_Status*) {}  // single queue: ordered
void SEGetStreamStatus(const SP_Device*, SP_Stream, TF_Status*) {}
void SECreateEvent(const SP_Device*, SP_Event* event, TF_Status*) { *event = (SP_Event)(void*)new EventObj(); }
void SEDestroyEvent(const SP_Device*, SP_Event event) { delete (EventObj*)(void*)event; }
SE_EventStatus SEGetEventStatus(const SP_Device*, SP_Event event) {
  auto* e = (EventObj*)(void*)event;
  return g.stream->completed_token() >= e->token ? SE_EVENT_COMPLETE : SE_EVENT_PENDING;
}
void SERecordEvent(const SP_Device*, SP_Stream, SP_Event event, TF_Status* status) {
  auto r = g.stream->RecordEvent();  // mandatory flush trigger (§6.3)
  if (!r.ok()) { TF_SetStatus(status, TF_INTERNAL, r.status.message.c_str()); return; }
  ((EventObj*)(void*)event)->token = r.value.token;
}
void SEWaitEvent(const SP_Device*, SP_Stream, SP_Event, TF_Status*) { g.stream->BlockHostUntilDone(); }
uint64_t NowNs() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
void SECreateTimer(const SP_Device*, SP_Timer* timer, TF_Status*) { *timer = (SP_Timer)(void*)new TimerObj(); }
void SEDestroyTimer(const SP_Device*, SP_Timer timer) { delete (TimerObj*)(void*)timer; }
void SEStartTimer(const SP_Device*, SP_Stream, SP_Timer timer, TF_Status*) { ((TimerObj*)(void*)timer)->start_ns = NowNs(); }
void SEStopTimer(const SP_Device*, SP_Stream, SP_Timer timer, TF_Status*) { ((TimerObj*)(void*)timer)->stop_ns = NowNs(); }
uint64_t SETimerNs(SP_Timer timer) {
  auto* t = (TimerObj*)(void*)timer;
  return t->stop_ns >= t->start_ns ? t->stop_ns - t->start_ns : 0;
}
tfvk::Allocation* AllocFromBase(const SP_DeviceMemoryBase* m) {
  if (m->payload) return (tfvk::Allocation*)(uintptr_t)m->payload;
  return Lookup(m->opaque);
}
void SEMemcpyDtoh(const SP_Device*, SP_Stream, void* dst, const SP_DeviceMemoryBase* src,
                  uint64_t size, TF_Status* status) {
  auto* a = AllocFromBase(src);
  if (!a) { TF_SetStatus(status, TF_INVALID_ARGUMENT, "tf-vulkan: unknown device buffer (dtoh)"); return; }
  auto s = g.stream->MemcpyDtoh(a, 0, dst, size, /*src_pending=*/true);
  if (!s.ok()) TF_SetStatus(status, TF_INTERNAL, s.message.c_str());
}
void SEMemcpyHtod(const SP_Device*, SP_Stream, SP_DeviceMemoryBase* dst, const void* src,
                  uint64_t size, TF_Status* status) {
  auto* a = AllocFromBase(dst);
  if (!a) { TF_SetStatus(status, TF_INVALID_ARGUMENT, "tf-vulkan: unknown device buffer (htod)"); return; }
  auto s = g.stream->MemcpyHtod(src, a, 0, size, /*dst_pending=*/true);
  if (!s.ok()) TF_SetStatus(status, TF_INTERNAL, s.message.c_str());
}
void SEMemcpyDtod(const SP_Device*, SP_Stream, SP_DeviceMemoryBase* dst,
                  const SP_DeviceMemoryBase* src, uint64_t size, TF_Status* status) {
  auto *a = AllocFromBase(src), *b = AllocFromBase(dst);
  if (!a || !b) { TF_SetStatus(status, TF_INVALID_ARGUMENT, "tf-vulkan: unknown device buffer (dtod)"); return; }
  // Same deferred stream: GPU copy when both are Vulkan buffers.
  auto s = g.kernels->Copy(a, b, size);
  if (!s.ok()) TF_SetStatus(status, TF_INTERNAL, s.message.c_str());
}
void SESyncDtoh(const SP_Device*, void* dst, const SP_DeviceMemoryBase* src, uint64_t size, TF_Status* st) {
  SEMemcpyDtoh(nullptr, nullptr, dst, src, size, st);
}
void SESyncHtod(const SP_Device*, SP_DeviceMemoryBase* dst, const void* src, uint64_t size, TF_Status* st) {
  SEMemcpyHtod(nullptr, nullptr, dst, src, size, st);
}
void SESyncDtod(const SP_Device*, SP_DeviceMemoryBase* dst, const SP_DeviceMemoryBase* src, uint64_t size, TF_Status* st) {
  SEMemcpyDtod(nullptr, nullptr, dst, src, size, st);
  g.stream->BlockHostUntilDone();
}
void SEBlockHostForEvent(const SP_Device*, SP_Event, TF_Status*) { g.stream->BlockHostUntilDone(); }
void SEBlockHostUntilDone(const SP_Device*, SP_Stream, TF_Status*) { g.stream->BlockHostUntilDone(); }
void SESyncAll(const SP_Device*, TF_Status*) { g.stream->BlockHostUntilDone(); }
void SEMemZero(const SP_Device*, SP_Stream, SP_DeviceMemoryBase* loc, uint64_t size, TF_Status*) {
  auto* a = AllocFromBase(loc);
  if (a && a->mapped) { g.stream->BlockHostUntilDone(); std::memset(a->mapped, 0, size); }
}
void SEMemset(const SP_Device*, SP_Stream, SP_DeviceMemoryBase* loc, uint8_t p, uint64_t size, TF_Status*) {
  auto* a = AllocFromBase(loc);
  if (a && a->mapped) { g.stream->BlockHostUntilDone(); std::memset(a->mapped, p, size); }
}
void SEMemset32(const SP_Device*, SP_Stream, SP_DeviceMemoryBase* loc, uint32_t p, uint64_t size, TF_Status*) {
  auto* a = AllocFromBase(loc);
  if (a && a->mapped) {
    g.stream->BlockHostUntilDone();
    auto* d = (uint32_t*)a->mapped;
    for (uint64_t i = 0; i < size / 4; ++i) d[i] = p;
  }
}
TF_Bool SEHostCallback(const SP_Device*, SP_Stream, SE_StatusCallbackFn fn, void* arg) {
  g.stream->BlockHostUntilDone();
  TF_Status* s = TF_NewStatus();
  fn(arg, s);
  TF_DeleteStatus(s);
  return 1;
}
void CreateStreamExecutor(const SP_Platform*, SE_CreateStreamExecutorParams* params, TF_Status* status) {
  if (!EnsureInit()) { TF_SetStatus(status, TF_UNAVAILABLE, "tf-vulkan: Vulkan init failed"); return; }
  SP_StreamExecutor* se = params->stream_executor;
  se->struct_size = SP_STREAMEXECUTOR_STRUCT_SIZE;
  se->ext = nullptr;
  se->allocate = SEAllocate; se->deallocate = SEDeallocate;
  se->host_memory_allocate = SEHostAlloc; se->host_memory_deallocate = SEHostDealloc;
  se->unified_memory_allocate = SEUnifiedAlloc; se->unified_memory_deallocate = SEUnifiedDealloc;
  se->get_allocator_stats = SEGetAllocStats; se->device_memory_usage = SEDeviceMemUsage;
  se->create_stream = SECreateStream; se->destroy_stream = SEDestroyStream;
  se->create_stream_dependency = SEStreamDep; se->get_stream_status = SEGetStreamStatus;
  se->create_event = SECreateEvent; se->destroy_event = SEDestroyEvent;
  se->get_event_status = SEGetEventStatus; se->record_event = SERecordEvent;
  se->wait_for_event = SEWaitEvent;
  se->create_timer = SECreateTimer; se->destroy_timer = SEDestroyTimer;
  se->start_timer = SEStartTimer; se->stop_timer = SEStopTimer;
  se->memcpy_dtoh = SEMemcpyDtoh; se->memcpy_htod = SEMemcpyHtod; se->memcpy_dtod = SEMemcpyDtod;
  se->sync_memcpy_dtoh = SESyncDtoh; se->sync_memcpy_htod = SESyncHtod; se->sync_memcpy_dtod = SESyncDtod;
  se->block_host_for_event = SEBlockHostForEvent; se->block_host_until_done = SEBlockHostUntilDone;
  se->synchronize_all_activity = SESyncAll;
  se->mem_zero = SEMemZero; se->memset = SEMemset; se->memset32 = SEMemset32;
  se->host_callback = SEHostCallback;
}
void DestroyStreamExecutor(const SP_Platform*, SP_StreamExecutor*) {}
void GetDeviceCount(const SP_Platform*, int* count, TF_Status*) { *count = EnsureInit() ? 1 : 0; }
void CreateDevice(const SP_Platform*, SE_CreateDeviceParams* params, TF_Status*) {
  EnsureInit();
  SP_Device* d = params->device;
  d->struct_size = SP_DEVICE_STRUCT_SIZE;
  d->ext = nullptr;
  d->ordinal = params->ordinal;
  d->device_handle = nullptr;
  d->hardware_name = g.hardware_name.c_str();
  d->device_vendor = "tf-vulkan";
  d->pci_bus_id = nullptr;
}
void DestroyDevice(const SP_Platform*, SP_Device*) {}
int32_t GetNumaNode(const SP_Device*) { return -1; }
int64_t GetBandwidth(const SP_Device*) { return -1; }
double GetGflops(const SP_Device*) { return -1; }
void CreateDeviceFns(const SP_Platform*, SE_CreateDeviceFnsParams* params, TF_Status*) {
  SP_DeviceFns* f = params->device_fns;
  f->struct_size = SP_DEVICE_FNS_STRUCT_SIZE;
  f->ext = nullptr;
  f->get_numa_node = GetNumaNode; f->get_memory_bandwidth = GetBandwidth; f->get_gflops = GetGflops;
}
void DestroyDeviceFns(const SP_Platform*, SP_DeviceFns*) {}
void CreateTimerFns(const SP_Platform*, SP_TimerFns* timer, TF_Status*) {
  timer->struct_size = SP_TIMER_FNS_STRUCT_SIZE;
  timer->ext = nullptr;
  timer->nanoseconds = SETimerNs;
}
void DestroyTimerFns(const SP_Platform*, SP_TimerFns*) {}
void DestroyPlatform(SP_Platform*) {}
void DestroyPlatformFns(SP_PlatformFns*) {}

// ---------------- Kernel helpers ----------------
void Fail(TF_OpKernelContext* ctx, TF_Code code, const char* msg) {
  TF_Status* s = TF_NewStatus();
  TF_SetStatus(s, code, msg);
  TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}
struct TensorGuard {
  TF_Tensor* t = nullptr;
  ~TensorGuard() { if (t) TF_DeleteTensor(t); }
};
bool GetInput(TF_OpKernelContext* ctx, int i, TensorGuard& out) {
  TF_Status* s = TF_NewStatus();
  TF_GetInput(ctx, i, &out.t, s);
  bool ok = TF_GetCode(s) == TF_OK;
  TF_DeleteStatus(s);
  return ok;
}
std::vector<int64_t> Dims(TF_Tensor* t) {
  std::vector<int64_t> d(TF_NumDims(t));
  for (size_t i = 0; i < d.size(); ++i) d[i] = TF_Dim(t, (int)i);
  return d;
}
int64_t Numel(TF_Tensor* t) {
  int64_t n = 1;
  for (int i = 0; i < TF_NumDims(t); ++i) n *= TF_Dim(t, i);
  return n;
}
// P0 submission policy at the TF boundary (§6.3): TF core kernels
// (assign, Identity, Reshape) read/write our UMA mapped memory directly
// without stream synchronization, so plugin computes must drain the
// deferred stream before returning or those kernels read stale data
// (verified: Keras optimizer updates silently no-op without this).
// The native runtime keeps full deferred submission; this sync is the
// TF-integration policy until hazard tracking exists at this layer.
struct SyncOnExit {
  ~SyncOnExit() { if (g.ready) g.stream->BlockHostUntilDone(); }
};
tfvk::Allocation* TensorAlloc(TF_Tensor* t, TF_OpKernelContext* ctx) {
  auto* a = Lookup(TF_TensorData(t));
  if (!a) Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: tensor is not in a mapped Vulkan allocation");
  return a;
}
TF_Tensor* AllocOutput(TF_OpKernelContext* ctx, const std::vector<int64_t>& dims,
                       TF_DataType dtype = TF_FLOAT) {
  TF_Status* s = TF_NewStatus();
  int64_t n = 1; for (auto d : dims) n *= d;
  TF_Tensor* t = TF_AllocateOutput(ctx, 0, dtype, dims.data(), (int)dims.size(),
                                   (size_t)n * (dtype == TF_HALF ? 2 : 4), s);
  bool ok = TF_GetCode(s) == TF_OK;
  TF_DeleteStatus(s);
  if (!ok) return nullptr;
  return t;
}
bool AttrBool(TF_OpKernelConstruction* c, const char* name) {
  TF_Status* s = TF_NewStatus();
  unsigned char v = 0;
  TF_OpKernelConstruction_GetAttrBool(c, name, &v, s);
  TF_DeleteStatus(s);
  return v != 0;
}
}  // namespace

namespace {
// ---------------- Kernel compute (FP32, §7 K0/K1) ----------------
struct MatMulCfg { bool ta = false, tb = false; };
void* MatMulCreate(TF_OpKernelConstruction* c) {
  auto* cfg = new MatMulCfg();
  // MatMul uses transpose_a/b; BatchMatMul(V2) uses adj_x/adj_y.
  cfg->ta = AttrBool(c, "transpose_a") || AttrBool(c, "adj_x");
  cfg->tb = AttrBool(c, "transpose_b") || AttrBool(c, "adj_y");
  return cfg;
}
void DeleteCfg(void* p) { delete (MatMulCfg*)p; }

void MatMulCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (MatMulCfg*)kernel;
  TensorGuard a, b;
  if (!GetInput(ctx, 0, a) || !GetInput(ctx, 1, b)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: MatMul inputs");
  TF_DataType mdt = TF_TensorType(a.t);
  if (mdt != TF_TensorType(b.t) || (mdt != TF_FLOAT && mdt != TF_HALF))
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: MatMul FP32/FP16 only");
  if (mdt == TF_HALF && !g.device.info().shader_float16)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: FP16 gated off (shaderFloat16, §6.4)");
  auto ad = Dims(a.t), bd = Dims(b.t);
  if (ad.size() < 2 || bd.size() < 2)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: MatMul rank < 2");
  // Last two dims are the matrix; leading dims are broadcast batch dims.
  int64_t M = cfg->ta ? ad[ad.size() - 1] : ad[ad.size() - 2];
  int64_t K1 = cfg->ta ? ad[ad.size() - 2] : ad[ad.size() - 1];
  int64_t K2 = cfg->tb ? bd[bd.size() - 1] : bd[bd.size() - 2];
  int64_t N = cfg->tb ? bd[bd.size() - 2] : bd[bd.size() - 1];
  if (K1 != K2) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: MatMul inner dims mismatch");
  std::vector<int64_t> ab(ad.begin(), ad.end() - 2), bb(bd.begin(), bd.end() - 2);
  size_t brank = std::max(ab.size(), bb.size());
  std::vector<int64_t> batch(brank, 1);
  auto batch_at = [](const std::vector<int64_t>& v, size_t brank_, size_t d) -> int64_t {
    if (d < brank_ - v.size()) return 1;
    return v[d - (brank_ - v.size())];
  };
  for (size_t d = 0; d < brank; ++d) {
    int64_t av = batch_at(ab, brank, d), bv = batch_at(bb, brank, d);
    if (av != bv && av != 1 && bv != 1)
      return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: MatMul batch dims not broadcastable");
    batch[d] = std::max(av, bv);
  }
  std::vector<int64_t> od = batch;
  od.push_back(M); od.push_back(N);
  TensorGuard out{AllocOutput(ctx, od, mdt)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: MatMul output alloc");
  auto *aa = TensorAlloc(a.t, ctx), *bb_alloc = TensorAlloc(b.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!aa || !bb_alloc || !cc) return;
  int64_t total_batch = 1; for (auto d : batch) total_batch *= d;
  auto batch_offset = [&](const std::vector<int64_t>& bdims, int64_t mat_numel, int64_t flat) {
    // Map flat output-batch index to this operand's batch element offset.
    int64_t off = 0, rem = flat;
    // strides over output batch dims
    std::vector<int64_t> ostr(batch.size(), 1);
    for (int i = (int)batch.size() - 2; i >= 0; --i) ostr[i] = ostr[i + 1] * batch[i + 1];
    std::vector<int64_t> istr(bdims.size(), 1);
    for (int i = (int)bdims.size() - 2; i >= 0; --i) istr[i] = istr[i + 1] * bdims[i + 1];
    for (size_t d = 0; d < batch.size(); ++d) {
      int64_t coord = ostr[d] ? rem / ostr[d] : 0;
      rem -= coord * ostr[d];
      int64_t dim = batch_at(bdims, brank, d);
      int64_t c = dim == 1 ? 0 : coord;
      size_t id = d - (brank - bdims.size());
      if (!bdims.empty() && d >= brank - bdims.size()) off += c * istr[id] * mat_numel;
    }
    return (uint32_t)off;
  };
  for (int64_t batch_idx = 0; batch_idx < total_batch; ++batch_idx) {
    uint32_t ao = batch_offset(ab, (cfg->ta ? K1 * M : M * K1), batch_idx);
    uint32_t bo = batch_offset(bb, (cfg->tb ? N * K2 : K2 * N), batch_idx);
    uint32_t co = (uint32_t)(batch_idx * M * N);
    auto s = mdt == TF_HALF
        ? g.kernels->MatMulF16(aa, bb_alloc, cc, (uint32_t)M, (uint32_t)N, (uint32_t)K1,
                               cfg->ta, cfg->tb, ao, bo, co)
        : g.kernels->MatMul(aa, bb_alloc, cc, (uint32_t)M, (uint32_t)N, (uint32_t)K1,
                            cfg->ta, cfg->tb, ao, bo, co);
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  }
  // TF_AllocateOutput already sets the output (TF 2.21); guard deletes the handle.
}

struct BinaryCfg { tfvk::BinaryOp op; };
template <tfvk::BinaryOp Op>
void* BinaryCreate(TF_OpKernelConstruction*) { return new BinaryCfg{Op}; }
void BinaryCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (BinaryCfg*)kernel;
  TensorGuard a, b;
  if (!GetInput(ctx, 0, a) || !GetInput(ctx, 1, b)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: binary inputs");
  TF_DataType bdt = TF_TensorType(a.t);
  if (bdt != TF_TensorType(b.t) || (bdt != TF_FLOAT && bdt != TF_HALF))
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: binary FP32/FP16 only");
  if (bdt == TF_HALF && !g.device.info().shader_float16)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: FP16 gated off (shaderFloat16, §6.4)");
  auto ad = Dims(a.t), bd = Dims(b.t);
  // numpy-style broadcast output shape.
  std::vector<int64_t> od(std::max(ad.size(), bd.size()), 1);
  for (size_t d = 0; d < od.size(); ++d) {
    int64_t av = d < od.size() - ad.size() ? 1 : ad[d - (od.size() - ad.size())];
    int64_t bv = d < od.size() - bd.size() ? 1 : bd[d - (od.size() - bd.size())];
    if (av != bv && av != 1 && bv != 1)
      return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: binary shapes not broadcastable");
    od[d] = std::max(av, bv);
  }
  if (od.size() > 4) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: binary rank > 4");
  TensorGuard out{AllocOutput(ctx, od, bdt)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: binary output alloc");
  auto *aa = TensorAlloc(a.t, ctx), *bb = TensorAlloc(b.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!aa || !bb || !cc) return;
  tfvk::Status s;
  if (bdt == TF_HALF) {
    int64_t na = Numel(a.t), nb = Numel(b.t);
    if (na != nb && nb != 1)
      return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: FP16 binary broadcast limited to scalar rhs");
    s = g.kernels->BinaryF16(cfg->op, aa, bb, cc, (uint32_t)na, nb == 1);
  } else {
    s = g.kernels->BinaryBroadcast(cfg->op, aa, bb, cc, od, ad, bd);
  }
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}
void DeleteBinary(void* p) { delete (BinaryCfg*)p; }

struct UnaryCfg { tfvk::UnaryOp op; };
template <tfvk::UnaryOp Op>
void* UnaryCreate(TF_OpKernelConstruction*) { return new UnaryCfg{Op}; }
void UnaryCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (UnaryCfg*)kernel;
  TensorGuard a;
  if (!GetInput(ctx, 0, a)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: unary input");
  TF_DataType udt = TF_TensorType(a.t);
  if (udt != TF_FLOAT && udt != TF_HALF) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: unary FP32/FP16 only");
  if (udt == TF_HALF && !g.device.info().shader_float16)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: FP16 gated off (shaderFloat16, §6.4)");
  TensorGuard out{AllocOutput(ctx, Dims(a.t), udt)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: unary output alloc");
  auto *aa = TensorAlloc(a.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!aa || !cc) return;
  auto s = udt == TF_HALF ? g.kernels->UnaryF16(cfg->op, aa, cc, (uint32_t)Numel(a.t))
                          : g.kernels->Unary(cfg->op, aa, cc, (uint32_t)Numel(a.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}
void DeleteUnary(void* p) { delete (UnaryCfg*)p; }

void BiasAddCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x, bias;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, bias)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: BiasAdd inputs");
  auto xd = Dims(x.t);
  if (xd.empty() || Numel(bias.t) != xd.back())
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: BiasAdd bias must match last dim");
  TF_DataType bdt2 = TF_TensorType(x.t);
  if (bdt2 != TF_FLOAT && bdt2 != TF_HALF) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: BiasAdd FP32/FP16 only");
  TensorGuard out{AllocOutput(ctx, xd, bdt2)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: BiasAdd output alloc");
  auto *xx = TensorAlloc(x.t, ctx), *bb = TensorAlloc(bias.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !bb || !cc) return;
  auto s = bdt2 == TF_HALF ? g.kernels->BiasAddF16(xx, bb, cc, (uint32_t)Numel(x.t), (uint32_t)Numel(bias.t))
                           : g.kernels->BiasAdd(xx, bb, cc, (uint32_t)Numel(x.t), (uint32_t)Numel(bias.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}

struct ReduceCfg { tfvk::ReduceOp op; bool keep_dims = false; };
void* SumCreate(TF_OpKernelConstruction* c) { return new ReduceCfg{tfvk::ReduceOp::kSum, AttrBool(c, "keep_dims")}; }
void* MeanCreate(TF_OpKernelConstruction* c) { return new ReduceCfg{tfvk::ReduceOp::kMean, AttrBool(c, "keep_dims")}; }
void DeleteReduce(void* p) { delete (ReduceCfg*)p; }
void ReduceCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (ReduceCfg*)kernel;
  TensorGuard x, idx;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, idx)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: reduce inputs");
  auto xd = Dims(x.t);
  int rank = (int)xd.size();
  std::vector<int> axes;
  int64_t nidx = Numel(idx.t);
  for (int64_t i = 0; i < nidx; ++i) {
    int64_t v = TF_TensorType(idx.t) == TF_INT64 ? ((int64_t*)TF_TensorData(idx.t))[i]
                                                 : ((int32_t*)TF_TensorData(idx.t))[i];
    if (v < 0) v += rank;
    axes.push_back((int)v);
  }
  if (axes.empty()) for (int i = 0; i < rank; ++i) axes.push_back(i);
  std::sort(axes.begin(), axes.end(), std::greater<int>());
  // Reduce one axis at a time on device allocations.
  auto* cur = TensorAlloc(x.t, ctx);
  if (!cur) return;
  std::vector<int64_t> cur_dims = xd;
  std::unique_ptr<tfvk::Allocation, decltype(&free)> tmp_guard(nullptr, &free);
  tfvk::Allocation* owned = nullptr;
  for (int axis : axes) {
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= cur_dims[i];
    for (size_t i = axis + 1; i < cur_dims.size(); ++i) inner *= cur_dims[i];
    auto r = g.allocator->Allocate((size_t)(outer * inner) * 4, g.stream->completed_token());
    if (!r.ok()) return Fail(ctx, TF_INTERNAL, "tf-vulkan: reduce temp alloc");
    RegisterAlloc(r.value);
    auto s = g.kernels->Reduce(cfg->op, cur, r.value, (uint32_t)outer, (uint32_t)cur_dims[axis], (uint32_t)inner);
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
    if (owned) { UnregisterAlloc(owned); g.allocator->Release(owned, g.stream->last_submitted_token()); }
    owned = r.value;
    cur = owned;
    cur_dims.erase(cur_dims.begin() + axis);
    if (cfg->keep_dims) cur_dims.insert(cur_dims.begin() + axis, 1);
    // NOTE: keep_dims changes logical shape only; storage layout identical.
    if (cfg->keep_dims) cur_dims.erase(cur_dims.begin() + axis), cur_dims.insert(cur_dims.begin() + axis, 1);
  }
  std::vector<int64_t> out_dims = xd;
  // Recompute output dims from original + axes + keep_dims.
  {
    std::vector<bool> reduced(rank, false);
    // axes were sorted descending and mutated; recompute from idx again:
    std::vector<int> ax2;
    for (int64_t i = 0; i < nidx; ++i) {
      int64_t v = TF_TensorType(idx.t) == TF_INT64 ? ((int64_t*)TF_TensorData(idx.t))[i] : ((int32_t*)TF_TensorData(idx.t))[i];
      if (v < 0) v += rank;
      ax2.push_back((int)v);
    }
    if (ax2.empty()) for (int i = 0; i < rank; ++i) ax2.push_back(i);
    for (int a : ax2) reduced[a] = true;
    out_dims.clear();
    for (int i = 0; i < rank; ++i) {
      if (!reduced[i]) out_dims.push_back(xd[i]);
      else if (cfg->keep_dims) out_dims.push_back(1);
    }
    if (out_dims.empty() && !cfg->keep_dims) out_dims = {};
  }
  TensorGuard out{AllocOutput(ctx, out_dims)};
  if (!out.t) { if (owned) { UnregisterAlloc(owned); g.allocator->Release(owned, 0); } return Fail(ctx, TF_INTERNAL, "tf-vulkan: reduce output alloc"); }
  auto* cc = TensorAlloc(out.t, ctx);
  if (!cc) { if (owned) { UnregisterAlloc(owned); g.allocator->Release(owned, 0); } return; }
  int64_t total = 1; for (auto d : out_dims) total *= d;
  if (out_dims.empty()) total = 1;
  auto s = g.kernels->Copy(cur, cc, (uint64_t)total * 4);
  if (owned) { UnregisterAlloc(owned); g.allocator->Release(owned, g.stream->last_submitted_token()); }
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}

void SoftmaxCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x;
  if (!GetInput(ctx, 0, x)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Softmax input");
  auto xd = Dims(x.t);
  if (xd.empty()) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Softmax scalar");
  int64_t cols = xd.back(), rows = Numel(x.t) / cols;
  TensorGuard out{AllocOutput(ctx, xd)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Softmax output alloc");
  auto *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !cc) return;
  auto s = g.kernels->Softmax(xx, cc, (uint32_t)rows, (uint32_t)cols);
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}

void IdentityCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x;
  if (!GetInput(ctx, 0, x)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Identity input");
  TensorGuard out{AllocOutput(ctx, Dims(x.t))};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Identity output alloc");
  auto *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !cc) return;
  auto s = (TF_TensorType(x.t) == TF_FLOAT)
      ? g.kernels->Unary(tfvk::UnaryOp::kIdentity, xx, cc, (uint32_t)Numel(x.t))
      : g.kernels->Copy(xx, cc, TF_TensorByteSize(x.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  // TF_AllocateOutput already sets the output (TF 2.21).
}

template <typename S>
bool CastFrom(const void* src, TF_DataType dt, void* dst, int64_t n) {
  auto* s = (const S*)src;
  switch (dt) {
    case TF_FLOAT: { auto* d = (float*)dst; for (int64_t i = 0; i < n; ++i) d[i] = (float)s[i]; return true; }
    case TF_DOUBLE: { auto* d = (double*)dst; for (int64_t i = 0; i < n; ++i) d[i] = (double)s[i]; return true; }
    case TF_INT32: { auto* d = (int32_t*)dst; for (int64_t i = 0; i < n; ++i) d[i] = (int32_t)s[i]; return true; }
    case TF_INT64: { auto* d = (int64_t*)dst; for (int64_t i = 0; i < n; ++i) d[i] = (int64_t)s[i]; return true; }
    case TF_BOOL: { auto* d = (bool*)dst; for (int64_t i = 0; i < n; ++i) d[i] = s[i] != 0; return true; }
    default: return false;
  }
}
void CastCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x;
  if (!GetInput(ctx, 0, x)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Cast input");
  TF_DataType dt = TF_ExpectedOutputDataType(ctx, 0);
  auto xd = Dims(x.t);
  int64_t n = Numel(x.t);
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, dt, xd.data(), (int)xd.size(),
                                       (size_t)n * TF_DataTypeSize(dt), st);
  bool ok = TF_GetCode(st) == TF_OK;
  TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Cast output alloc");
  TensorGuard out{out_t};
  auto *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !cc) return;
  if (TF_TensorType(x.t) == TF_FLOAT && dt == TF_FLOAT) {
    auto s = g.kernels->Unary(tfvk::UnaryOp::kIdentity, xx, cc, (uint32_t)n);
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
    return;
  }
  if (TF_TensorType(x.t) == TF_FLOAT && dt == TF_HALF) {
    auto s = g.kernels->CastToF16(xx, cc, (uint32_t)n);
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
    return;
  }
  if (TF_TensorType(x.t) == TF_HALF && dt == TF_FLOAT) {
    auto s = g.kernels->CastFromF16(xx, cc, (uint32_t)n);
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
    return;
  }
  // Numeric cast on UMA mapped memory (plumbing op, §7 K0): sync, convert
  // in place through the mapped pointers, no host round-trip allocation.
  g.stream->BlockHostUntilDone();
  const void* src = TF_TensorData(x.t);
  void* dst = TF_TensorData(out.t);
  bool done = false;
  switch (TF_TensorType(x.t)) {
    case TF_FLOAT: done = CastFrom<float>(src, dt, dst, n); break;
    case TF_DOUBLE: done = CastFrom<double>(src, dt, dst, n); break;
    case TF_INT32: done = CastFrom<int32_t>(src, dt, dst, n); break;
    case TF_INT64: done = CastFrom<int64_t>(src, dt, dst, n); break;
    case TF_BOOL: done = CastFrom<bool>(src, dt, dst, n); break;
    default: break;
  }
  if (!done) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Cast dtype unsupported");
}

// ---- Plumbing ops on mapped UMA memory (§7 K0): tiny metadata/random ops
// that unblock strict-placement graph construction. Not compute kernels.
void FloorModCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard a, b;
  if (!GetInput(ctx, 0, a) || !GetInput(ctx, 1, b)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: FloorMod inputs");
  TF_DataType dt = TF_TensorType(a.t);
  TensorGuard out;
  {
    TF_Status* st = TF_NewStatus();
    auto d = Dims(a.t);
    out.t = TF_AllocateOutput(ctx, 0, dt, d.data(), (int)d.size(), TF_TensorByteSize(a.t), st);
    bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
    if (!ok) return Fail(ctx, TF_INTERNAL, "tf-vulkan: FloorMod alloc");
  }
  g.stream->BlockHostUntilDone();
  int64_t n = Numel(a.t), nb = Numel(b.t);
  if (dt == TF_INT32) {
    auto *x = (int32_t*)TF_TensorData(a.t), *y = (int32_t*)TF_TensorData(b.t), *o = (int32_t*)TF_TensorData(out.t);
    for (int64_t i = 0; i < n; ++i) { int32_t d = y[nb == 1 ? 0 : i]; int32_t r = d ? x[i] % d : x[i]; o[i] = (r != 0 && ((r < 0) != (d < 0))) ? r + d : r; }
  } else if (dt == TF_INT64) {
    auto *x = (int64_t*)TF_TensorData(a.t), *y = (int64_t*)TF_TensorData(b.t), *o = (int64_t*)TF_TensorData(out.t);
    for (int64_t i = 0; i < n; ++i) { int64_t d = y[nb == 1 ? 0 : i]; int64_t r = d ? x[i] % d : x[i]; o[i] = (r != 0 && ((r < 0) != (d < 0))) ? r + d : r; }
  } else if (dt == TF_FLOAT) {
    auto *x = (float*)TF_TensorData(a.t), *y = (float*)TF_TensorData(b.t), *o = (float*)TF_TensorData(out.t);
    for (int64_t i = 0; i < n; ++i) { float d = y[nb == 1 ? 0 : i]; o[i] = d == 0 ? x[i] : x[i] - std::floor(x[i] / d) * d; }
  } else return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: FloorMod dtype");
}
struct RandomCfg { bool normal = false; int64_t seed = 0, seed2 = 0; };
void* RandomCreate(TF_OpKernelConstruction* c, bool normal) {
  auto* cfg = new RandomCfg();
  cfg->normal = normal;
  TF_Status* s = TF_NewStatus();
  int32_t v = 0;
  TF_OpKernelConstruction_GetAttrInt32(c, "seed", &v, s); cfg->seed = v;
  TF_OpKernelConstruction_GetAttrInt32(c, "seed2", &v, s); cfg->seed2 = v;
  TF_DeleteStatus(s);
  return cfg;
}
void* RandomUniformCreate(TF_OpKernelConstruction* c) { return RandomCreate(c, false); }
void* RandomNormalCreate(TF_OpKernelConstruction* c) { return RandomCreate(c, true); }
void DeleteRandom(void* p) { delete (RandomCfg*)p; }
void RandomCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (RandomCfg*)kernel;
  TensorGuard shape;
  if (!GetInput(ctx, 0, shape)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: random shape");
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < Numel(shape.t); ++i)
    dims.push_back(TF_TensorType(shape.t) == TF_INT64 ? ((int64_t*)TF_TensorData(shape.t))[i]
                                                      : ((int32_t*)TF_TensorData(shape.t))[i]);
  TF_DataType dt = TF_ExpectedOutputDataType(ctx, 0);
  if (dt != TF_FLOAT) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: random FP32 only");
  int64_t n = 1; for (auto d : dims) n *= d;
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims.data(), (int)dims.size(), (size_t)n * 4, st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: random alloc");
  TensorGuard out{out_t};
  if (!TensorAlloc(out.t, ctx)) return;
  g.stream->BlockHostUntilDone();
  std::mt19937 rng((uint32_t)(cfg->seed * 1000003 + cfg->seed2 + 1));
  float* o = (float*)TF_TensorData(out.t);
  if (cfg->normal) {
    std::normal_distribution<float> nd;
    for (int64_t i = 0; i < n; ++i) o[i] = nd(rng);
  } else {
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    for (int64_t i = 0; i < n; ++i) o[i] = ud(rng);
  }
}

// Stateless random plumbing (Keras 3 initializers, §7 K0): deterministic
// Philox-free MT fill seeded from key/counter; values only, not TF's exact
// Philox stream — documented divergence, sufficient for initialization.
void StatelessGetKeyCounterCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard seed;
  if (!GetInput(ctx, 0, seed)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: seed");
  auto* s = (int64_t*)TF_TensorData(seed.t);
  TF_Status* st = TF_NewStatus();
  int64_t one = 1, two = 2;
  TF_Tensor* key = TF_AllocateOutput(ctx, 0, TF_UINT64, &one, 1, 8, st);
  if (TF_GetCode(st) != TF_OK) { TF_DeleteStatus(st); return Fail(ctx, TF_INTERNAL, "tf-vulkan: key alloc"); }
  TF_Tensor* ctr = TF_AllocateOutput(ctx, 1, TF_UINT64, &two, 1, 16, st);
  if (TF_GetCode(st) != TF_OK) { TF_DeleteStatus(st); return Fail(ctx, TF_INTERNAL, "tf-vulkan: counter alloc"); }
  // TF 2.21: StatelessRandomGetKeyCounter has 2 outputs (key, counter).
  TensorGuard kg{key}, cg{ctr};
  g.stream->BlockHostUntilDone();
  ((uint64_t*)TF_TensorData(key))[0] = (uint64_t)s[0];
  ((uint64_t*)TF_TensorData(ctr))[0] = (uint64_t)s[1];
  ((uint64_t*)TF_TensorData(ctr))[1] = 0;
}
struct StatelessCfg { bool normal = false; };
void* StatelessUniformCreate(TF_OpKernelConstruction*) { return new StatelessCfg{false}; }
void* StatelessNormalCreate(TF_OpKernelConstruction*) { return new StatelessCfg{true}; }
void DeleteStateless(void* p) { delete (StatelessCfg*)p; }
void StatelessRandomCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (StatelessCfg*)kernel;
  TensorGuard shape, key, ctr;
  if (!GetInput(ctx, 0, shape) || !GetInput(ctx, 1, key) || !GetInput(ctx, 2, ctr))
    return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: stateless random inputs");
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < Numel(shape.t); ++i)
    dims.push_back(TF_TensorType(shape.t) == TF_INT64 ? ((int64_t*)TF_TensorData(shape.t))[i]
                                                      : ((int32_t*)TF_TensorData(shape.t))[i]);
  int64_t n = 1; for (auto d : dims) n *= d;
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, TF_FLOAT, dims.data(), (int)dims.size(), (size_t)n * 4, st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: stateless random alloc");
  TensorGuard out{out_t};
  if (!TensorAlloc(out.t, ctx)) return;
  g.stream->BlockHostUntilDone();
  uint64_t k = ((uint64_t*)TF_TensorData(key.t))[0];
  uint64_t c = ((uint64_t*)TF_TensorData(ctr.t))[0];
  std::mt19937 rng((uint32_t)(k * 2654435761u + c + 7));
  float* o = (float*)TF_TensorData(out.t);
  if (cfg->normal) { std::normal_distribution<float> nd; for (int64_t i = 0; i < n; ++i) o[i] = nd(rng); }
  else { std::uniform_real_distribution<float> ud(0.0f, 1.0f); for (int64_t i = 0; i < n; ++i) o[i] = ud(rng); }
}

void FillCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard dims_t, val;
  if (!GetInput(ctx, 0, dims_t) || !GetInput(ctx, 1, val)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Fill inputs");
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < Numel(dims_t.t); ++i)
    dims.push_back(TF_TensorType(dims_t.t) == TF_INT64 ? ((int64_t*)TF_TensorData(dims_t.t))[i]
                                                       : ((int32_t*)TF_TensorData(dims_t.t))[i]);
  TF_DataType dt = TF_TensorType(val.t);
  int64_t n = 1; for (auto d : dims) n *= d;
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, dt, dims.data(), (int)dims.size(),
                                       (size_t)n * TF_DataTypeSize(dt), st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Fill alloc");
  TensorGuard out{out_t};
  if (!TensorAlloc(out.t, ctx)) return;
  g.stream->BlockHostUntilDone();
  void* dst = TF_TensorData(out.t);
  const void* src = TF_TensorData(val.t);
  if (dt == TF_FLOAT) { float v = *(float*)src; auto* d = (float*)dst; for (int64_t i = 0; i < n; ++i) d[i] = v; }
  else if (dt == TF_INT32) { int32_t v = *(int32_t*)src; auto* d = (int32_t*)dst; for (int64_t i = 0; i < n; ++i) d[i] = v; }
  else if (dt == TF_INT64) { int64_t v = *(int64_t*)src; auto* d = (int64_t*)dst; for (int64_t i = 0; i < n; ++i) d[i] = v; }
  else if (dt == TF_BOOL) { bool v = *(bool*)src; auto* d = (bool*)dst; for (int64_t i = 0; i < n; ++i) d[i] = v; }
  else return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Fill dtype");
}
void ReshapeCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x, shape_t;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, shape_t)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Reshape inputs");
  std::vector<int64_t> dims;
  int64_t known = 1, infer = -1;
  for (int64_t i = 0; i < Numel(shape_t.t); ++i) {
    int64_t v = TF_TensorType(shape_t.t) == TF_INT64 ? ((int64_t*)TF_TensorData(shape_t.t))[i]
                                                     : ((int32_t*)TF_TensorData(shape_t.t))[i];
    if (v == -1) infer = (int64_t)dims.size();
    else known *= v;
    dims.push_back(v);
  }
  if (infer >= 0) dims[infer] = Numel(x.t) / known;
  TF_DataType dt = TF_TensorType(x.t);
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, dt, dims.data(), (int)dims.size(), TF_TensorByteSize(x.t), st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Reshape alloc");
  TensorGuard out{out_t};
  auto *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !cc) return;
  auto s = g.kernels->Copy(xx, cc, TF_TensorByteSize(x.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
}

void TileCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard x, mult;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, mult)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Tile inputs");
  auto xd = Dims(x.t);
  std::vector<int64_t> od;
  for (size_t i = 0; i < xd.size(); ++i) {
    int64_t m = TF_TensorType(mult.t) == TF_INT64 ? ((int64_t*)TF_TensorData(mult.t))[i]
                                                  : ((int32_t*)TF_TensorData(mult.t))[i];
    od.push_back(xd[i] * m);
  }
  TF_DataType dt = TF_TensorType(x.t);
  int64_t n = 1; for (auto d : od) n *= d;
  TF_Status* st = TF_NewStatus();
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, dt, od.data(), (int)od.size(),
                                       (size_t)n * TF_DataTypeSize(dt), st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Tile alloc");
  TensorGuard out{out_t};
  if (!TensorAlloc(out.t, ctx) || !TensorAlloc(x.t, ctx)) return;
  g.stream->BlockHostUntilDone();
  // Mapped-memory tile (gradient plumbing, §10): index remap per element.
  auto istr = [&] { std::vector<int64_t> s(xd.size(), 1); for (int i = (int)xd.size()-2; i >= 0; --i) s[i]=s[i+1]*xd[i+1]; return s; }();
  auto ostr = [&] { std::vector<int64_t> s(od.size(), 1); for (int i = (int)od.size()-2; i >= 0; --i) s[i]=s[i+1]*od[i+1]; return s; }();
  const float* src = (const float*)TF_TensorData(x.t);
  float* dst = (float*)TF_TensorData(out.t);
  if (dt != TF_FLOAT) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Tile FP32 only");
  for (int64_t i = 0; i < n; ++i) {
    int64_t rem = i, src_idx = 0;
    for (size_t d = 0; d < od.size(); ++d) {
      int64_t coord = ostr[d] ? rem / ostr[d] : 0;
      rem -= coord * ostr[d];
      src_idx += (coord % xd[d]) * istr[d];
    }
    dst[i] = src[src_idx];
  }
}

void AddNCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  int n_in = TF_NumInputs(ctx);
  if (n_in < 1) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: AddN no inputs");
  TensorGuard first;
  if (!GetInput(ctx, 0, first)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: AddN input");
  auto dims = Dims(first.t);
  TensorGuard out{AllocOutput(ctx, dims)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: AddN alloc");
  auto* acc = TensorAlloc(out.t, ctx);
  auto* a0 = TensorAlloc(first.t, ctx);
  if (!acc || !a0) return;
  auto s = g.kernels->Copy(a0, acc, TF_TensorByteSize(first.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  for (int i = 1; i < n_in; ++i) {
    TensorGuard in;
    if (!GetInput(ctx, i, in)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: AddN input");
    auto* ai = TensorAlloc(in.t, ctx);
    if (!ai) return;
    auto r = g.allocator->Allocate(TF_TensorByteSize(first.t), g.stream->completed_token());
    if (!r.ok()) return Fail(ctx, TF_INTERNAL, "tf-vulkan: AddN temp");
    RegisterAlloc(r.value);
    s = g.kernels->BinaryBroadcast(tfvk::BinaryOp::kAdd, acc, ai, r.value, dims, dims, dims);
    if (s.ok()) s = g.kernels->Copy(r.value, acc, TF_TensorByteSize(first.t));
    UnregisterAlloc(r.value);
    g.allocator->Release(r.value, g.stream->last_submitted_token());
    if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
  }
}

void ReluGradCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard grad, x;
  if (!GetInput(ctx, 0, grad) || !GetInput(ctx, 1, x)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: ReluGrad inputs");
  TensorGuard out{AllocOutput(ctx, Dims(x.t))};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: ReluGrad alloc");
  auto *gg = TensorAlloc(grad.t, ctx), *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!gg || !xx || !cc) return;
  auto s = g.kernels->ReluGrad(gg, xx, cc, (uint32_t)Numel(x.t));
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
}
void BiasAddGradCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  TensorGuard grad;
  if (!GetInput(ctx, 0, grad)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: BiasAddGrad input");
  auto gd = Dims(grad.t);
  if (gd.empty()) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: BiasAddGrad scalar");
  int64_t channels = gd.back(), rows = Numel(grad.t) / channels;
  TensorGuard out{AllocOutput(ctx, {channels})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: BiasAddGrad alloc");
  auto *gg = TensorAlloc(grad.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!gg || !cc) return;
  auto s = g.kernels->Reduce(tfvk::ReduceOp::kSum, gg, cc, 1, (uint32_t)rows, (uint32_t)channels);
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
}

template <typename T>
void IntAdd(const TensorGuard& a, const TensorGuard& b, const TensorGuard& out) {
  const T* x = (const T*)TF_TensorData(a.t);
  const T* y = (const T*)TF_TensorData(b.t);
  T* o = (T*)TF_TensorData(out.t);
  int64_t n = Numel(a.t), nb = Numel(b.t);
  for (int64_t i = 0; i < n; ++i) o[i] = x[i] + y[nb == 1 ? 0 : i];
}
void AddIntCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  // Integer AddV2 for gradient shape plumbing (§10) on mapped UMA memory.
  TensorGuard a, b;
  if (!GetInput(ctx, 0, a) || !GetInput(ctx, 1, b)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: int add inputs");
  if (Dims(a.t) != Dims(b.t) && Numel(b.t) != 1)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: int add broadcast");
  TF_DataType dt = TF_TensorType(a.t);
  TF_Status* st = TF_NewStatus();
  auto d = Dims(a.t);
  TF_Tensor* out_t = TF_AllocateOutput(ctx, 0, dt, d.data(), (int)d.size(), TF_TensorByteSize(a.t), st);
  bool ok = TF_GetCode(st) == TF_OK; TF_DeleteStatus(st);
  if (!ok || !out_t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: int add alloc");
  TensorGuard out{out_t};
  if (!TensorAlloc(out.t, ctx)) return;
  g.stream->BlockHostUntilDone();
  if (dt == TF_INT32) IntAdd<int32_t>(a, b, out);
  else if (dt == TF_INT64) IntAdd<int64_t>(a, b, out);
  else return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: int add dtype");
}

struct ConvCfg {
  int32_t strides[4] = {1, 1, 1, 1};
  int32_t dilations[4] = {1, 1, 1, 1};
  char padding[16] = "VALID";
  char format[8] = "NHWC";
};
void* ConvCreate(TF_OpKernelConstruction* c) {
  auto* cfg = new ConvCfg();
  TF_Status* s = TF_NewStatus();
  int32_t vals[4];
  TF_OpKernelConstruction_GetAttrInt32List(c, "strides", vals, 4, s);
  if (TF_GetCode(s) == TF_OK) memcpy(cfg->strides, vals, sizeof(vals));
  TF_OpKernelConstruction_GetAttrInt32List(c, "dilations", vals, 4, s);
  if (TF_GetCode(s) == TF_OK) memcpy(cfg->dilations, vals, sizeof(vals));
  char buf[32] = {0};
  TF_OpKernelConstruction_GetAttrString(c, "padding", buf, sizeof(buf), s);
  if (TF_GetCode(s) == TF_OK) std::strncpy(cfg->padding, buf, sizeof(cfg->padding) - 1);
  memset(buf, 0, sizeof(buf));
  TF_OpKernelConstruction_GetAttrString(c, "data_format", buf, sizeof(buf), s);
  if (TF_GetCode(s) == TF_OK) std::strncpy(cfg->format, buf, sizeof(cfg->format) - 1);
  TF_DeleteStatus(s);
  return cfg;
}
void DeleteConv(void* p) { delete (ConvCfg*)p; }
void Conv2DCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (ConvCfg*)kernel;
  if (std::strcmp(cfg->format, "NHWC") != 0)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Conv2D NHWC only");
  if (cfg->dilations[1] != 1 || cfg->dilations[2] != 1)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Conv2D dilation 1 only");
  TensorGuard x, flt;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, flt)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Conv2D inputs");
  auto xd = Dims(x.t), fd = Dims(flt.t);
  if (xd.size() != 4 || fd.size() != 4) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Conv2D rank 4");
  int64_t B=xd[0], H=xd[1], W=xd[2], Cin=xd[3];
  int64_t Kh=fd[0], Kw=fd[1], Cout=fd[3];
  if (fd[2] != Cin) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Conv2D Cin mismatch");
  int64_t sH=cfg->strides[1], sW=cfg->strides[2], outH, outW, padH=0, padW=0;
  if (std::strcmp(cfg->padding, "SAME") == 0) {
    outH = (H + sH - 1) / sH; outW = (W + sW - 1) / sW;
    padH = std::max<int64_t>(0, (outH - 1) * sH + Kh - H) / 2;
    padW = std::max<int64_t>(0, (outW - 1) * sW + Kw - W) / 2;
  } else {
    outH = (H - Kh) / sH + 1; outW = (W - Kw) / sW + 1;
  }
  TensorGuard out{AllocOutput(ctx, {B, outH, outW, Cout})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Conv2D alloc");
  auto *xx = TensorAlloc(x.t, ctx), *ff = TensorAlloc(flt.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !ff || !cc) return;
  auto s = g.kernels->Conv2D(xx, ff, cc, (uint32_t)B, (uint32_t)H, (uint32_t)W, (uint32_t)Cin,
                             (uint32_t)Cout, (uint32_t)Kh, (uint32_t)Kw, (uint32_t)sH, (uint32_t)sW,
                             (uint32_t)padH, (uint32_t)padW, (uint32_t)outH, (uint32_t)outW);
  if (!s.ok()) return Fail(ctx, TF_INTERNAL, s.message.c_str());
}

struct PoolCfg {
  int32_t ksize[4] = {1, 1, 1, 1};
  int32_t strides[4] = {1, 1, 1, 1};
  char padding[16] = "VALID";
  bool avg = false;
};
void* PoolCreateImpl(TF_OpKernelConstruction* c, bool avg) {
  auto* cfg = new PoolCfg();
  cfg->avg = avg;
  TF_Status* st = TF_NewStatus();
  int32_t vals[4];
  TF_OpKernelConstruction_GetAttrInt32List(c, "ksize", vals, 4, st);
  if (TF_GetCode(st) == TF_OK) memcpy(cfg->ksize, vals, sizeof(vals));
  TF_OpKernelConstruction_GetAttrInt32List(c, "strides", vals, 4, st);
  if (TF_GetCode(st) == TF_OK) memcpy(cfg->strides, vals, sizeof(vals));
  char buf[32] = {0};
  TF_OpKernelConstruction_GetAttrString(c, "padding", buf, sizeof(buf), st);
  if (TF_GetCode(st) == TF_OK) std::strncpy(cfg->padding, buf, sizeof(cfg->padding) - 1);
  TF_DeleteStatus(st);
  return cfg;
}
void* MaxPoolCreate(TF_OpKernelConstruction* c) { return PoolCreateImpl(c, false); }
void* AvgPoolCreate(TF_OpKernelConstruction* c) { return PoolCreateImpl(c, true); }
void DeletePool(void* p) { delete (PoolCfg*)p; }
void PoolCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (PoolCfg*)kernel;
  TensorGuard x;
  if (!GetInput(ctx, 0, x)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Pool input");
  auto xd = Dims(x.t);
  if (xd.size() != 4) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Pool NHWC rank4");
  int64_t B=xd[0],H=xd[1],W=xd[2],C=xd[3];
  int64_t Kh=cfg->ksize[1],Kw=cfg->ksize[2],sH=cfg->strides[1],sW=cfg->strides[2];
  int64_t outH,outW,padH=0,padW=0;
  if (std::strcmp(cfg->padding,"SAME")==0) {
    outH=(H+sH-1)/sH; outW=(W+sW-1)/sW;
    padH=std::max<int64_t>(0,(outH-1)*sH+Kh-H)/2; padW=std::max<int64_t>(0,(outW-1)*sW+Kw-W)/2;
  } else { outH=(H-Kh)/sH+1; outW=(W-Kw)/sW+1; }
  TensorGuard out{AllocOutput(ctx, {B,outH,outW,C})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Pool alloc");
  auto *xx=TensorAlloc(x.t,ctx), *cc=TensorAlloc(out.t,ctx);
  if (!xx||!cc) return;
  auto st = g.kernels->Pool(cfg->avg, xx, cc, (uint32_t)B,(uint32_t)H,(uint32_t)W,(uint32_t)C,
      (uint32_t)Kh,(uint32_t)Kw,(uint32_t)sH,(uint32_t)sW,(uint32_t)padH,(uint32_t)padW,
      (uint32_t)outH,(uint32_t)outW);
  if (!st.ok()) return Fail(ctx, TF_INTERNAL, st.message.c_str());
}
void DepthwiseCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (ConvCfg*)kernel;
  if (cfg->dilations[1] != 1 || cfg->dilations[2] != 1)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Depthwise dilation 1 only");
  TensorGuard x, flt;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, flt)) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: Depthwise inputs");
  auto xd=Dims(x.t), fd=Dims(flt.t);
  if (xd.size()!=4||fd.size()!=4) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: Depthwise rank4");
  int64_t B=xd[0],H=xd[1],W=xd[2],Cin=xd[3],Kh=fd[0],Kw=fd[1],Mult=fd[3];
  int64_t sH=cfg->strides[1],sW=cfg->strides[2],outH,outW,padH=0,padW=0;
  if (std::strcmp(cfg->padding,"SAME")==0) {
    outH=(H+sH-1)/sH; outW=(W+sW-1)/sW;
    padH=std::max<int64_t>(0,(outH-1)*sH+Kh-H)/2; padW=std::max<int64_t>(0,(outW-1)*sW+Kw-W)/2;
  } else { outH=(H-Kh)/sH+1; outW=(W-Kw)/sW+1; }
  TensorGuard out{AllocOutput(ctx, {B,outH,outW,Cin*Mult})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: Depthwise alloc");
  auto *xx=TensorAlloc(x.t,ctx), *ff=TensorAlloc(flt.t,ctx), *cc=TensorAlloc(out.t,ctx);
  if (!xx||!ff||!cc) return;
  auto st = g.kernels->DepthwiseConv(xx, ff, cc, (uint32_t)B,(uint32_t)H,(uint32_t)W,(uint32_t)Cin,
      (uint32_t)Mult,(uint32_t)Kh,(uint32_t)Kw,(uint32_t)sH,(uint32_t)sW,(uint32_t)padH,(uint32_t)padW,
      (uint32_t)outH,(uint32_t)outW);
  if (!st.ok()) return Fail(ctx, TF_INTERNAL, st.message.c_str());
}
// Conv2D geometry from forward params, shared by backprop ops.
bool ConvGeometry(ConvCfg* cfg, int64_t H, int64_t W, int64_t Kh, int64_t Kw,
                  int64_t* outH, int64_t* outW, int64_t* padH, int64_t* padW) {
  int64_t sH=cfg->strides[1], sW=cfg->strides[2];
  if (std::strcmp(cfg->padding,"SAME")==0) {
    *outH=(H+sH-1)/sH; *outW=(W+sW-1)/sW;
    *padH=std::max<int64_t>(0,(*outH-1)*sH+Kh-H)/2; *padW=std::max<int64_t>(0,(*outW-1)*sW+Kw-W)/2;
  } else { *outH=(H-Kh)/sH+1; *outW=(W-Kw)/sW+1; *padH=0; *padW=0; }
  return true;
}
void ConvBackpropFilterCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (ConvCfg*)kernel;
  TensorGuard x, fshape, grad;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, fshape) || !GetInput(ctx, 2, grad))
    return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: ConvBackpropFilter inputs");
  auto xd=Dims(x.t), gd=Dims(grad.t);
  int64_t fs[4];
  for (int i=0;i<4;++i) fs[i]=((int32_t*)TF_TensorData(fshape.t))[i];
  int64_t B=xd[0],H=xd[1],W=xd[2],Cin=xd[3],Kh=fs[0],Kw=fs[1],Cout=fs[3];
  int64_t outH,outW,padH,padW;
  ConvGeometry(cfg,H,W,Kh,Kw,&outH,&outW,&padH,&padW);
  if (gd[1]!=outH||gd[2]!=outW) return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: backprop grad shape");
  TensorGuard out{AllocOutput(ctx, {Kh,Kw,Cin,Cout})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: backprop filter alloc");
  auto *xx=TensorAlloc(x.t,ctx), *gg=TensorAlloc(grad.t,ctx), *cc=TensorAlloc(out.t,ctx);
  if (!xx||!gg||!cc) return;
  auto st=g.kernels->ConvFilterGrad(xx,gg,cc,(uint32_t)B,(uint32_t)H,(uint32_t)W,(uint32_t)Cin,
      (uint32_t)Cout,(uint32_t)Kh,(uint32_t)Kw,(uint32_t)cfg->strides[1],(uint32_t)cfg->strides[2],
      (uint32_t)padH,(uint32_t)padW,(uint32_t)outH,(uint32_t)outW);
  if (!st.ok()) return Fail(ctx, TF_INTERNAL, st.message.c_str());
}
void ConvBackpropInputCompute(void* kernel, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto* cfg = (ConvCfg*)kernel;
  TensorGuard ishape, flt, grad;
  if (!GetInput(ctx, 0, ishape) || !GetInput(ctx, 1, flt) || !GetInput(ctx, 2, grad))
    return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: ConvBackpropInput inputs");
  auto fd=Dims(flt.t), gd=Dims(grad.t);
  int64_t is[4];
  for (int i=0;i<4;++i) is[i]= TF_TensorType(ishape.t)==TF_INT64 ? ((int64_t*)TF_TensorData(ishape.t))[i] : ((int32_t*)TF_TensorData(ishape.t))[i];
  int64_t B=is[0],H=is[1],W=is[2],Cin=is[3],Kh=fd[0],Kw=fd[1],Cout=fd[3];
  int64_t outH,outW,padH,padW;
  ConvGeometry(cfg,H,W,Kh,Kw,&outH,&outW,&padH,&padW);
  TensorGuard out{AllocOutput(ctx, {B,H,W,Cin})};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: backprop input alloc");
  auto *gg=TensorAlloc(grad.t,ctx), *ff=TensorAlloc(flt.t,ctx), *cc=TensorAlloc(out.t,ctx);
  if (!gg||!ff||!cc) return;
  auto st=g.kernels->ConvInputGrad(gg,ff,cc,(uint32_t)B,(uint32_t)H,(uint32_t)W,(uint32_t)Cin,
      (uint32_t)Cout,(uint32_t)Kh,(uint32_t)Kw,(uint32_t)cfg->strides[1],(uint32_t)cfg->strides[2],
      (uint32_t)padH,(uint32_t)padW,(uint32_t)outH,(uint32_t)outW);
  if (!st.ok()) return Fail(ctx, TF_INTERNAL, st.message.c_str());
  (void)gd;
}

void BroadcastToCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync2;
  TensorGuard x, shape_t;
  if (!GetInput(ctx, 0, x) || !GetInput(ctx, 1, shape_t))
    return Fail(ctx, TF_INVALID_ARGUMENT, "tf-vulkan: BroadcastTo inputs");
  std::vector<int64_t> od;
  for (int64_t i = 0; i < Numel(shape_t.t); ++i)
    od.push_back(TF_TensorType(shape_t.t) == TF_INT64 ? ((int64_t*)TF_TensorData(shape_t.t))[i]
                                                      : ((int32_t*)TF_TensorData(shape_t.t))[i]);
  if (TF_TensorType(x.t) != TF_FLOAT)
    return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: BroadcastTo FP32 only");
  TensorGuard out{AllocOutput(ctx, od)};
  if (!out.t) return Fail(ctx, TF_INTERNAL, "tf-vulkan: BroadcastTo alloc");
  auto *xx = TensorAlloc(x.t, ctx), *cc = TensorAlloc(out.t, ctx);
  if (!xx || !cc) return;
  // out = x + 0, with the zero scalar broadcast alongside x's own broadcast.
  auto zr = g.allocator->Allocate(4, g.stream->completed_token());
  if (!zr.ok()) return Fail(ctx, TF_INTERNAL, "tf-vulkan: BroadcastTo zero alloc");
  RegisterAlloc(zr.value);
  g.stream->BlockHostUntilDone();
  *(float*)zr.value->mapped = 0.0f;
  auto st = g.kernels->BinaryBroadcast(tfvk::BinaryOp::kAdd, xx, zr.value, cc, od, Dims(x.t),
                                       std::vector<int64_t>{1});
  UnregisterAlloc(zr.value);
  g.allocator->Release(zr.value, g.stream->last_submitted_token());
  if (!st.ok()) return Fail(ctx, TF_INTERNAL, st.message.c_str());
}

void CopyTensor(TF_OpKernelContext* ctx, TF_Tensor* src, TF_Tensor* dst) {
  auto *a = Lookup(TF_TensorData(src)), *b = Lookup(TF_TensorData(dst));
  size_t n = TF_TensorByteSize(src);
  if (a && b) { g.kernels->Copy(a, b, n); return; }
  // Host-visible fallback for variable plumbing tensors (still mapped UMA in
  // the normal path; this covers TF-owned host tensors).
  g.stream->BlockHostUntilDone();
  std::memcpy(TF_TensorData(dst), TF_TensorData(src), n);
  (void)ctx;
}
// TF_AssignVariable is in kernels_experimental.h but is NOT exported by
// every TF build (verified absent in the TF 2.21 macOS wheel, F1 §5).
// Resolve it at runtime; skip registration when the pinned build lacks it.
using AssignVariableFn = void (*)(TF_OpKernelContext*, int, int, bool,
                                  void (*)(TF_OpKernelContext*, TF_Tensor*, TF_Tensor*),
                                  TF_Status*);
AssignVariableFn ResolveAssignVariable() {
  return (AssignVariableFn)dlsym(RTLD_DEFAULT, "TF_AssignVariable");
}
void AssignVariableCompute(void*, TF_OpKernelContext* ctx) {
  SyncOnExit _tfvk_sync;
  auto fn = ResolveAssignVariable();
  if (!fn) return Fail(ctx, TF_UNIMPLEMENTED, "tf-vulkan: TF_AssignVariable not exported by this TF build (F1)");
  TF_Status* s = TF_NewStatus();
  fn(ctx, 0, 1, /*validate_shape=*/false, CopyTensor, s);
  if (TF_GetCode(s) != TF_OK) TF_OpKernelContext_Failure(ctx, s);
  TF_DeleteStatus(s);
}

void Register(const char* op, void* (*create)(TF_OpKernelConstruction*),
              void (*compute)(void*, TF_OpKernelContext*), void (*del)(void*),
              const char* host_input = nullptr) {
  TF_Status* s = TF_NewStatus();
  TF_KernelBuilder* b = TF_NewKernelBuilder(op, DeviceType(), create, compute, del);
  TF_KernelBuilder_TypeConstraint(b, "T", TF_FLOAT, s);
  if (host_input) TF_KernelBuilder_HostMemory(b, host_input);
  TF_RegisterKernelBuilder(op, b, s);
  TF_DeleteStatus(s);
}
}  // namespace

extern "C" void TF_InitKernel() {
  if (!EnsureInit()) return;
  Register("MatMul", MatMulCreate, MatMulCompute, DeleteCfg);
  Register("BatchMatMulV2", MatMulCreate, MatMulCompute, DeleteCfg);
  Register("BatchMatMul", MatMulCreate, MatMulCompute, DeleteCfg);
  Register("AddV2", BinaryCreate<tfvk::BinaryOp::kAdd>, BinaryCompute, DeleteBinary);
  Register("Add", BinaryCreate<tfvk::BinaryOp::kAdd>, BinaryCompute, DeleteBinary);
  Register("Sub", BinaryCreate<tfvk::BinaryOp::kSub>, BinaryCompute, DeleteBinary);
  Register("Mul", BinaryCreate<tfvk::BinaryOp::kMul>, BinaryCompute, DeleteBinary);
  Register("Div", BinaryCreate<tfvk::BinaryOp::kDiv>, BinaryCompute, DeleteBinary);
  Register("RealDiv", BinaryCreate<tfvk::BinaryOp::kDiv>, BinaryCompute, DeleteBinary);
  Register("DivNoNan", BinaryCreate<tfvk::BinaryOp::kDivNoNan>, BinaryCompute, DeleteBinary);
  Register("Pow", BinaryCreate<tfvk::BinaryOp::kPow>, BinaryCompute, DeleteBinary);
  Register("Maximum", BinaryCreate<tfvk::BinaryOp::kMaximum>, BinaryCompute, DeleteBinary);
  Register("SquaredDifference", BinaryCreate<tfvk::BinaryOp::kSquaredDiff>, BinaryCompute, DeleteBinary);
  Register("Relu", UnaryCreate<tfvk::UnaryOp::kRelu>, UnaryCompute, DeleteUnary);
  Register("Relu6", UnaryCreate<tfvk::UnaryOp::kRelu6>, UnaryCompute, DeleteUnary);
  Register("Sigmoid", UnaryCreate<tfvk::UnaryOp::kSigmoid>, UnaryCompute, DeleteUnary);
  Register("Tanh", UnaryCreate<tfvk::UnaryOp::kTanh>, UnaryCompute, DeleteUnary);
  Register("Exp", UnaryCreate<tfvk::UnaryOp::kExp>, UnaryCompute, DeleteUnary);
  Register("Log", UnaryCreate<tfvk::UnaryOp::kLog>, UnaryCompute, DeleteUnary);
  Register("Sqrt", UnaryCreate<tfvk::UnaryOp::kSqrt>, UnaryCompute, DeleteUnary);
  Register("Neg", UnaryCreate<tfvk::UnaryOp::kNeg>, UnaryCompute, DeleteUnary);
  Register("Square", UnaryCreate<tfvk::UnaryOp::kSquare>, UnaryCompute, DeleteUnary);
  Register("Conv2D", ConvCreate, Conv2DCompute, DeleteConv);
  Register("DepthwiseConv2dNative", ConvCreate, DepthwiseCompute, DeleteConv);
  Register("Conv2DBackpropFilter", ConvCreate, ConvBackpropFilterCompute, DeleteConv);
  Register("Conv2DBackpropInput", ConvCreate, ConvBackpropInputCompute, DeleteConv);
  {
    TF_Status* st2 = TF_NewStatus();
    TF_KernelBuilder* mp = TF_NewKernelBuilder("MaxPool", DeviceType(), MaxPoolCreate, PoolCompute, DeletePool);
    TF_KernelBuilder_TypeConstraint(mp, "T", TF_FLOAT, st2);
    TF_RegisterKernelBuilder("MaxPool", mp, st2);
    TF_KernelBuilder* ap = TF_NewKernelBuilder("AvgPool", DeviceType(), AvgPoolCreate, PoolCompute, DeletePool);
    TF_KernelBuilder_TypeConstraint(ap, "T", TF_FLOAT, st2);
    TF_RegisterKernelBuilder("AvgPool", ap, st2);
    TF_DeleteStatus(st2);
  }
  Register("ReluGrad", nullptr, ReluGradCompute, nullptr);
  Register("BiasAddGrad", nullptr, BiasAddGradCompute, nullptr);
  // Identity is NOT registered: TF core's DEVICE_DEFAULT Identity matches
  // GPU at the same priority and duplicate registration is an error
  // (verified on TF 2.21). The default kernel operates on our mapped
  // device pointers, so residency is preserved (§7 K0 plumbing note).
  Register("BiasAdd", nullptr, BiasAddCompute, nullptr);
  Register("Sum", SumCreate, ReduceCompute, DeleteReduce, "reduction_indices");
  Register("Mean", MeanCreate, ReduceCompute, DeleteReduce, "reduction_indices");
  Register("Softmax", nullptr, SoftmaxCompute, nullptr);
  {
    TF_Status* s = TF_NewStatus();
    TF_KernelBuilder* b = TF_NewKernelBuilder("Cast", DeviceType(), nullptr, CastCompute, nullptr);
    TF_RegisterKernelBuilder("Cast", b, s);
    TF_KernelBuilder* fm = TF_NewKernelBuilder("FloorMod", DeviceType(), nullptr, FloorModCompute, nullptr);
    TF_RegisterKernelBuilder("FloorMod", fm, s);
    TF_KernelBuilder* ru = TF_NewKernelBuilder("RandomUniform", DeviceType(), RandomUniformCreate, RandomCompute, DeleteRandom);
    TF_KernelBuilder_HostMemory(ru, "shape");
    TF_RegisterKernelBuilder("RandomUniform", ru, s);
    TF_KernelBuilder* rn = TF_NewKernelBuilder("RandomStandardNormal", DeviceType(), RandomNormalCreate, RandomCompute, DeleteRandom);
    TF_KernelBuilder_HostMemory(rn, "shape");
    TF_RegisterKernelBuilder("RandomStandardNormal", rn, s);
    TF_KernelBuilder* gkc = TF_NewKernelBuilder("StatelessRandomGetKeyCounter", DeviceType(), nullptr, StatelessGetKeyCounterCompute, nullptr);
    TF_KernelBuilder_HostMemory(gkc, "seed");
    TF_RegisterKernelBuilder("StatelessRandomGetKeyCounter", gkc, s);
    TF_KernelBuilder* su = TF_NewKernelBuilder("StatelessRandomUniformV2", DeviceType(), StatelessUniformCreate, StatelessRandomCompute, DeleteStateless);
    TF_KernelBuilder_HostMemory(su, "shape"); TF_KernelBuilder_HostMemory(su, "key");
    TF_KernelBuilder_HostMemory(su, "counter"); TF_KernelBuilder_HostMemory(su, "alg");
    TF_RegisterKernelBuilder("StatelessRandomUniformV2", su, s);
    TF_KernelBuilder* iadd = TF_NewKernelBuilder("AddV2", DeviceType(), nullptr, AddIntCompute, nullptr);
    TF_KernelBuilder_TypeConstraint(iadd, "T", TF_INT32, s);
    TF_RegisterKernelBuilder("AddV2", iadd, s);
    TF_KernelBuilder* iadd64 = TF_NewKernelBuilder("AddV2", DeviceType(), nullptr, AddIntCompute, nullptr);
    TF_KernelBuilder_TypeConstraint(iadd64, "T", TF_INT64, s);
    TF_RegisterKernelBuilder("AddV2", iadd64, s);
    TF_KernelBuilder* addn = TF_NewKernelBuilder("AddN", DeviceType(), nullptr, AddNCompute, nullptr);
    TF_RegisterKernelBuilder("AddN", addn, s);
    TF_KernelBuilder* bto = TF_NewKernelBuilder("BroadcastTo", DeviceType(), nullptr, BroadcastToCompute, nullptr);
    TF_KernelBuilder_HostMemory(bto, "shape");
    TF_RegisterKernelBuilder("BroadcastTo", bto, s);
    TF_KernelBuilder* tile = TF_NewKernelBuilder("Tile", DeviceType(), nullptr, TileCompute, nullptr);
    TF_KernelBuilder_HostMemory(tile, "multiples");
    TF_RegisterKernelBuilder("Tile", tile, s);
    TF_KernelBuilder* fill = TF_NewKernelBuilder("Fill", DeviceType(), nullptr, FillCompute, nullptr);
    TF_KernelBuilder_HostMemory(fill, "dims");
    TF_RegisterKernelBuilder("Fill", fill, s);
    // Reshape is NOT registered: TF core already registers a GPU Reshape
    // at the same priority (verified TF 2.21), like Identity above.
    TF_KernelBuilder* sn = TF_NewKernelBuilder("StatelessRandomNormalV2", DeviceType(), StatelessNormalCreate, StatelessRandomCompute, DeleteStateless);
    TF_KernelBuilder_HostMemory(sn, "shape"); TF_KernelBuilder_HostMemory(sn, "key");
    TF_KernelBuilder_HostMemory(sn, "counter"); TF_KernelBuilder_HostMemory(sn, "alg");
    TF_RegisterKernelBuilder("StatelessRandomNormalV2", sn, s);
    TF_DeleteStatus(s);
  }
  // FP16 builders (K1.5): same compute functions, TF_HALF constraint.
  if (g.device.info().shader_float16) {
    auto reg_half = [&](const char* op, void* (*create)(TF_OpKernelConstruction*),
                        void (*compute)(void*, TF_OpKernelContext*), void (*del)(void*)) {
      TF_Status* st = TF_NewStatus();
      TF_KernelBuilder* b = TF_NewKernelBuilder(op, DeviceType(), create, compute, del);
      TF_KernelBuilder_TypeConstraint(b, "T", TF_HALF, st);
      TF_RegisterKernelBuilder(op, b, st);
      TF_DeleteStatus(st);
    };
    reg_half("MatMul", MatMulCreate, MatMulCompute, DeleteCfg);
    reg_half("BatchMatMulV2", MatMulCreate, MatMulCompute, DeleteCfg);
    reg_half("BatchMatMul", MatMulCreate, MatMulCompute, DeleteCfg);
    reg_half("AddV2", BinaryCreate<tfvk::BinaryOp::kAdd>, BinaryCompute, DeleteBinary);
    reg_half("Sub", BinaryCreate<tfvk::BinaryOp::kSub>, BinaryCompute, DeleteBinary);
    reg_half("Mul", BinaryCreate<tfvk::BinaryOp::kMul>, BinaryCompute, DeleteBinary);
    reg_half("Div", BinaryCreate<tfvk::BinaryOp::kDiv>, BinaryCompute, DeleteBinary);
    reg_half("RealDiv", BinaryCreate<tfvk::BinaryOp::kDiv>, BinaryCompute, DeleteBinary);
    reg_half("Relu", UnaryCreate<tfvk::UnaryOp::kRelu>, UnaryCompute, DeleteUnary);
    reg_half("Sigmoid", UnaryCreate<tfvk::UnaryOp::kSigmoid>, UnaryCompute, DeleteUnary);
    reg_half("Tanh", UnaryCreate<tfvk::UnaryOp::kTanh>, UnaryCompute, DeleteUnary);
    reg_half("BiasAdd", nullptr, BiasAddCompute, nullptr);
  }
  // K0.5 variable residency (§7): AssignVariableOp via kernels_experimental.h,
  // only when the pinned TF build actually exports TF_AssignVariable (F1).
  if (ResolveAssignVariable()) {
    TF_Status* s = TF_NewStatus();
    TF_KernelBuilder* b = TF_NewKernelBuilder("AssignVariableOp", DeviceType(), nullptr,
                                              AssignVariableCompute, nullptr);
    TF_RegisterKernelBuilder("AssignVariableOp", b, s);
    TF_DeleteStatus(s);
  }
}

namespace {
// ---- Grappler spike (task 010 / gate F7, spec §8) ----
// Pass-through optimizer: output graph is byte-identical (semantics
// preserved) while we record that TF invoked us, the graph size, and
// which registered pure ops appear in the serialized GraphDef. This
// proves the interception route exists on the pinned build; rewriting
// into _VulkanCompiledRegion stays gated on IREE execution (F6).
void GraphOptimize(void*, const TF_Buffer* in, const TF_GrapplerItem*, TF_Buffer* out,
                   TF_Status* status) {
  if (const char* log = std::getenv("TFVK_GRAPH_LOG")) {
    if (FILE* f = std::fopen(log, "a")) {
      std::string bytes((const char*)in->data, in->length);
      auto count_op = [&](const char* op) {
        size_t n = 0, pos = 0;
        std::string needle = op;
        while ((pos = bytes.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
        return n;
      };
      std::fprintf(f, "graph bytes=%zu MatMul=%zu Relu=%zu BiasAdd=%zu Conv2D=%zu Softmax=%zu\n",
                   in->length, count_op("MatMul"), count_op("Relu"), count_op("BiasAdd"),
                   count_op("Conv2D"), count_op("Softmax"));
      std::fclose(f);
    }
  }
  void* copy = std::malloc(in->length ? in->length : 1);
  std::memcpy(copy, in->data, in->length);
  out->data = copy;
  out->length = in->length;
  out->data_deallocator = [](void* data, size_t) { std::free(data); };
  TF_SetStatus(status, TF_OK, "");
}
void* GraphCreate() { return nullptr; }
void GraphDestroy(void*) {}
}  // namespace

extern "C" void TF_InitGraph(TP_OptimizerRegistrationParams* params, TF_Status* status) {
  params->device_type = DeviceType();
  TP_OptimizerConfigs* configs = params->optimizer_configs;
  configs->struct_size = TP_OPTIMIZER_CONFIGS_STRUCT_SIZE;
  configs->ext = nullptr;
  TP_Optimizer* opt = params->optimizer;
  opt->struct_size = TP_OPTIMIZER_STRUCT_SIZE;
  opt->ext = nullptr;
  opt->create_func = GraphCreate;
  opt->optimize_func = GraphOptimize;
  opt->destroy_func = GraphDestroy;
  TF_SetStatus(status, TF_OK, "");
}

extern "C" void SE_InitPlugin(SE_PlatformRegistrationParams* params, TF_Status* status) {
  params->major_version = SE_MAJOR; params->minor_version = SE_MINOR; params->patch_version = SE_PATCH;
  SP_Platform* p = params->platform;
  p->struct_size = SP_PLATFORM_STRUCT_SIZE;
  p->ext = nullptr;
  p->name = "VULKAN";            // subtype (§5, D1)
  p->type = DeviceType();        // device type (§5, D1)
  p->supports_unified_memory = 1;  // UMA mapped buffers (§6.2)
  p->use_bfc_allocator = 0;
  p->force_memory_growth = 1;
  SP_PlatformFns* f = params->platform_fns;
  f->struct_size = SP_PLATFORM_FNS_STRUCT_SIZE;
  f->ext = nullptr;
  f->get_device_count = GetDeviceCount;
  f->create_device = CreateDevice; f->destroy_device = DestroyDevice;
  f->create_device_fns = CreateDeviceFns; f->destroy_device_fns = DestroyDeviceFns;
  f->create_stream_executor = CreateStreamExecutor; f->destroy_stream_executor = DestroyStreamExecutor;
  f->create_timer_fns = CreateTimerFns; f->destroy_timer_fns = DestroyTimerFns;
  params->destroy_platform = DestroyPlatform;
  params->destroy_platform_fns = DestroyPlatformFns;
  if (!EnsureInit()) TF_SetStatus(status, TF_UNAVAILABLE, "tf-vulkan: no Vulkan compute device (Tier C)");
}
