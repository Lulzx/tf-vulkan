// Conformance tests (spec §15): overflow checks, allocator budget/recycling,
// deferred-submission flush triggers, MatMul vs float64 reference (odd shapes),
// pointwise/reduction/softmax parity. Vulkan tests run when a device exists.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "kernels/kernels.h"
#include "runtime/allocator.h"
#include "runtime/stream.h"
#include "runtime/tensor.h"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

static bool Close(float a, double b, double tol) { return std::abs(a - b) <= tol; }

void TestTensorOverflow() {
  tfvk::TensorDesc d{tfvk::DType::kF32, {INT64_MAX, 4}};
  CHECK(!d.numel().ok());
  tfvk::TensorDesc neg{tfvk::DType::kF32, {-1, 4}};
  CHECK(!neg.byte_size().ok());
  tfvk::TensorDesc ok{tfvk::DType::kF32, {7, 13, 5}};
  CHECK(ok.numel().value == 455);
  CHECK(ok.byte_size().value == 455 * 4);
  CHECK(!tfvk::CheckRange(100, 90, 20).ok());
  CHECK(tfvk::CheckRange(100, 90, 10).ok());
  CHECK(tfvk::ComputeBudgetBytes(1000, false, 0) == 300);       // 30% fallback (§6.1)
  CHECK(tfvk::ComputeBudgetBytes(1000, true, 700) == 700);       // VK_EXT_memory_budget
}

void TestCpuReference() {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd;
  for (auto [M, K, N] : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {7, 13, 5}, {33, 17, 29}}) {
    std::vector<float> a(M * K), b(K * N), c(M * N);
    std::vector<double> ad(M * K), bd(K * N), cd(M * N);
    for (size_t i = 0; i < a.size(); ++i) { a[i] = nd(rng); ad[i] = a[i]; }
    for (size_t i = 0; i < b.size(); ++i) { b[i] = nd(rng); bd[i] = b[i]; }
    tfvk::CpuMatMul(a.data(), b.data(), c.data(), M, N, K);
    tfvk::CpuMatMulF64(ad.data(), bd.data(), cd.data(), M, N, K);
    double mx = 0; for (double v : cd) mx = std::max(mx, std::abs(v));
    double tol = 1e-5 * std::sqrt((double)K) * mx;
    for (size_t i = 0; i < c.size(); ++i) CHECK(Close(c[i], cd[i], tol));
  }
  CHECK(tfvk::CpuUnary(tfvk::UnaryOp::kRelu, -2) == 0);
  CHECK(tfvk::CpuBinary(tfvk::BinaryOp::kDivNoNan, 1, 0) == 0);
}

#ifdef TFVK_HAS_VULKAN
struct Fixture {
  tfvk::VulkanDevice dev;
  std::unique_ptr<tfvk::Allocator> alloc;
  std::unique_ptr<tfvk::Stream> stream;
  tfvk::VulkanKernels kernels;
  Fixture() : kernels(&dev, nullptr) {}
  bool Init() {
    if (!dev.Init("").ok()) return false;
    uint64_t heap = dev.info().heap_sizes.empty() ? 0 : dev.info().heap_sizes[0];
    alloc.reset(new tfvk::Allocator(&dev, tfvk::ComputeBudgetBytes(heap, dev.info().memory_budget_ext, dev.info().budget_heap_size)));
    stream.reset(new tfvk::Stream(&dev, alloc.get()));
    kernels = tfvk::VulkanKernels(&dev, stream.get());
    return kernels.Init().ok();
  }
  tfvk::Allocation* Buf(const std::vector<float>& v) {
    auto r = alloc->Allocate(v.size() * 4, stream->completed_token());
    if (!r.ok()) return nullptr;
    stream->MemcpyHtod(v.data(), r.value, 0, v.size() * 4, false);
    return r.value;
  }
  std::vector<float> Read(tfvk::Allocation* a, size_t n, bool pending) {
    std::vector<float> out(n);
    stream->MemcpyDtoh(a, 0, out.data(), n * 4, pending);
    return out;
  }
};

void TestVulkanMatMulOddShapes(Fixture& f) {
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd;
  // P0 shapes from §15 (1024^3 covered in bench; keep test time bounded).
  for (auto [M, K, N] : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {7, 13, 5}, {63, 65, 31}, {128, 128, 128}}) {
    std::vector<float> av(M * K), bv(K * N);
    for (auto& x : av) x = nd(rng);
    for (auto& x : bv) x = nd(rng);
    auto *a = f.Buf(av), *b = f.Buf(bv);
    auto cr = f.alloc->Allocate(M * N * 4, f.stream->completed_token());
    CHECK(cr.ok());
    CHECK(f.kernels.MatMul(a, b, cr.value, (uint32_t)M, (uint32_t)N, (uint32_t)K).ok());
    auto got = f.Read(cr.value, M * N, true);
    std::vector<double> ad(av.begin(), av.end()), bd(bv.begin(), bv.end()), cd(M * N);
    tfvk::CpuMatMulF64(ad.data(), bd.data(), cd.data(), M, N, K);
    double mx = 0; for (double v : cd) mx = std::max(mx, std::abs(v));
    double tol = 1e-5 * std::sqrt((double)K) * std::max(mx, 1.0);
    for (size_t i = 0; i < got.size(); ++i) CHECK(Close(got[i], cd[i], tol));
    f.alloc->Release(a, f.stream->last_submitted_token());
    f.alloc->Release(b, f.stream->last_submitted_token());
    f.alloc->Release(cr.value, f.stream->last_submitted_token());
  }
}

void TestVulkanChainAndFlush(Fixture& f) {
  // Eager chain MatMul -> BiasAdd -> Relu stays resident (§6.3, P1); dtoh flushes.
  std::vector<float> av(16, 0.5f), bv(16, 2.0f), bias(4, -1.0f);
  auto *a = f.Buf(av), *b = f.Buf(bv), *bs = f.Buf(bias);
  auto r1 = f.alloc->Allocate(16 * 4, 0); auto r2 = f.alloc->Allocate(16 * 4, 0); auto r3 = f.alloc->Allocate(16 * 4, 0);
  uint64_t subs_before = f.stream->stats().submissions;
  CHECK(f.kernels.MatMul(a, b, r1.value, 4, 4, 4).ok());
  CHECK(f.kernels.BiasAdd(r1.value, bs, r2.value, 16, 4).ok());
  CHECK(f.kernels.Unary(tfvk::UnaryOp::kRelu, r2.value, r3.value, 16).ok());
  CHECK(f.stream->stats().submissions == subs_before);  // deferred: no submit yet (§6.3)
  auto ev = f.stream->RecordEvent();
  CHECK(ev.ok());
  CHECK(f.stream->stats().flushes_event >= 1);  // record_event flush trigger (§6.3)
  auto got = f.Read(r3.value, 16, true);
  for (float v : got) CHECK(Close(v, 3.0, 1e-5));  // (4*0.5*2 - 1) relu = 3
  // Pointwise / reduction / softmax parity.
  std::vector<float> xv{1, 2, 3, 4, 5, 6};
  auto* x = f.Buf(xv);
  auto sr = f.alloc->Allocate(2 * 4, 0);
  CHECK(f.kernels.Reduce(tfvk::ReduceOp::kSum, x, sr.value, 2, 3, 1).ok());
  auto sgot = f.Read(sr.value, 2, true);
  CHECK(Close(sgot[0], 6, 1e-5)); CHECK(Close(sgot[1], 15, 1e-5));
  auto smr = f.alloc->Allocate(6 * 4, 0);
  CHECK(f.kernels.Softmax(x, smr.value, 2, 3).ok());
  auto sm = f.Read(smr.value, 6, true);
  for (int row = 0; row < 2; ++row) {
    std::vector<float> ref(3);
    tfvk::CpuSoftmaxRow(xv.data() + row * 3, ref.data(), 3);
    for (int i = 0; i < 3; ++i) CHECK(Close(sm[row * 3 + i], ref[i], 1e-5));
  }
}

void TestVulkanPointwiseAndThreshold(Fixture& f) {
  std::vector<float> xv{1, -2, 3, -4}, yv{0.5f, 2, -1, 4};
  auto *x = f.Buf(xv), *y = f.Buf(yv);
  auto r = f.alloc->Allocate(4 * 4, 0);
  CHECK(f.kernels.Binary(tfvk::BinaryOp::kAdd, x, y, r.value, 4, false).ok());
  auto got = f.Read(r.value, 4, true);
  for (int i = 0; i < 4; ++i) CHECK(Close(got[i], tfvk::CpuBinary(tfvk::BinaryOp::kAdd, xv[i], yv[i]), 1e-6));
  CHECK(f.kernels.Unary(tfvk::UnaryOp::kSigmoid, x, r.value, 4).ok());
  got = f.Read(r.value, 4, true);
  for (int i = 0; i < 4; ++i) CHECK(Close(got[i], tfvk::CpuUnary(tfvk::UnaryOp::kSigmoid, xv[i]), 1e-6));
  // Dispatch-count flush threshold (64, §6.3): 70 deferred unary dispatches.
  uint64_t thr_before = f.stream->stats().flushes_dispatch_threshold;
  for (int i = 0; i < 70; ++i) CHECK(f.kernels.Unary(tfvk::UnaryOp::kIdentity, x, r.value, 4).ok());
  CHECK(f.stream->stats().flushes_dispatch_threshold > thr_before);
  CHECK(f.stream->BlockHostUntilDone().ok());
}

void TestVulkanConv2D(Fixture& f) {
  // NHWC 1x5x5x3, filter 3x3x3x4, stride 1, SAME padding.
  const int B=1,H=5,W=5,Cin=3,Cout=4,Kh=3,Kw=3;
  std::mt19937 rng(3); std::normal_distribution<float> nd;
  std::vector<float> xv(B*H*W*Cin), fv(Kh*Kw*Cin*Cout);
  for (auto& v: xv) v = nd(rng);
  for (auto& v: fv) v = nd(rng);
  auto *x = f.Buf(xv), *flt = f.Buf(fv);
  auto yr = f.alloc->Allocate(B*H*W*Cout*4, 0);
  CHECK(yr.ok());
  CHECK(f.kernels.Conv2D(x, flt, yr.value, B,H,W,Cin,Cout,Kh,Kw,1,1,1,1,H,W).ok());
  auto got = f.Read(yr.value, B*H*W*Cout, true);
  std::vector<float> ref(B*H*W*Cout);
  tfvk::CpuConv2D(xv.data(), fv.data(), ref.data(), B,H,W,Cin,Cout,Kh,Kw,1,1,1,1,H,W);
  for (size_t i=0;i<got.size();++i) CHECK(Close(got[i], ref[i], 1e-4));
}

static uint16_t F2H(float f32) {
  uint32_t x; memcpy(&x, &f32, 4);
  uint32_t sign = (x >> 16) & 0x8000, exp = ((x >> 23) & 0xff) - 112, mant = x & 0x7fffff;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}
static float H2F(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ff;
  uint32_t x;
  if (exp == 0) x = sign;
  else if (exp == 31) x = sign | 0x7f800000;
  else x = sign | ((exp + 112) << 23) | (mant << 13);
  float f; memcpy(&f, &x, 4); return f;
}
void TestVulkanFp16(Fixture& f) {
  if (!f.dev.info().shader_float16) { printf("SKIP FP16 (gate off)\n"); return; }
  std::mt19937 rng(5); std::normal_distribution<float> nd;
  const int M=16,K=16,N=16;
  std::vector<float> av(M*K), bv(K*N);
  for (auto& v: av) v = nd(rng) * 0.5f;
  for (auto& v: bv) v = nd(rng) * 0.5f;
  std::vector<uint16_t> ah(M*K), bh(K*N);
  for (size_t i=0;i<av.size();++i) ah[i]=F2H(av[i]);
  for (size_t i=0;i<bv.size();++i) bh[i]=F2H(bv[i]);
  auto mk=[&](const std::vector<uint16_t>& v){
    auto r=f.alloc->Allocate(v.size()*2, f.stream->completed_token());
    if (!r.ok()) return (tfvk::Allocation*)nullptr;
    f.stream->MemcpyHtod(v.data(), r.value, 0, v.size()*2, false);
    return r.value;
  };
  auto *a=mk(ah), *b=mk(bh);
  auto cr=f.alloc->Allocate(M*N*2, 0);
  CHECK(cr.ok());
  CHECK(f.kernels.MatMulF16(a,b,cr.value,M,N,K).ok());
  std::vector<uint16_t> got(M*N);
  f.stream->MemcpyDtoh(cr.value, 0, got.data(), got.size()*2, true);
  std::vector<float> ref(M*N);
  tfvk::CpuMatMul(av.data(), bv.data(), ref.data(), M, N, K);
  for (size_t i=0;i<got.size();++i) CHECK(std::abs(H2F(got[i]) - ref[i]) < 0.05);
  // Cast roundtrip f32 -> f16 -> f32.
  auto *xf = f.Buf(av);
  auto hr = f.alloc->Allocate(av.size()*2, 0);
  auto fr = f.alloc->Allocate(av.size()*4, 0);
  CHECK(f.kernels.CastToF16(xf, hr.value, (uint32_t)av.size()).ok());
  CHECK(f.kernels.CastFromF16(hr.value, fr.value, (uint32_t)av.size()).ok());
  auto back = f.Read(fr.value, av.size(), true);
  for (size_t i=0;i<av.size();++i) CHECK(std::abs(back[i]-av[i]) < 0.01);
  printf("FP16 matmul+cast OK\n");
}

void TestAllocatorBudgetAndStability(Fixture& f) {
  tfvk::Allocator small(&f.dev, 8192);
  auto a = small.Allocate(4096, 0);
  CHECK(a.ok());
  auto b = small.Allocate(8192, 0);
  CHECK(!b.ok());  // budget enforced, explicit error not silent CPU (§6.2)
  // 10k alloc/release cycles without leak growth (bounded pool, §15 P0 gate scaled).
  uint64_t in_use_before = f.alloc->stats().bytes_in_use;
  for (int i = 0; i < 10000; ++i) {
    auto r = f.alloc->Allocate(1024, f.stream->completed_token());
    CHECK(r.ok());
    f.alloc->Release(r.value, f.stream->last_submitted_token());
  }
  CHECK(f.alloc->stats().bytes_in_use == in_use_before);
}
#endif

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);  // line-visible progress on device runs
  TestTensorOverflow();
  TestCpuReference();
#ifdef TFVK_HAS_VULKAN
  Fixture f;
  if (f.Init()) {
    printf("Vulkan device: %s (tier %d)\n", f.dev.info().name.c_str(), (int)f.dev.info().tier);
    printf("phase: MatMulOddShapes\n");
    TestVulkanMatMulOddShapes(f);
    printf("phase: ChainAndFlush\n");
    TestVulkanChainAndFlush(f);
    printf("phase: PointwiseAndThreshold\n");
    TestVulkanPointwiseAndThreshold(f);
    printf("phase: Conv2D\n");
    TestVulkanConv2D(f);
    printf("phase: Fp16\n");
    TestVulkanFp16(f);
    printf("phase: AllocatorBudgetAndStability\n");
    TestAllocatorBudgetAndStability(f);
    // Repeat dispatch stability (scaled-down 10k-cycle evidence).
    auto s = f.stream->BlockHostUntilDone();
    CHECK(s.ok());
    printf("stream: submissions=%llu dispatches=%llu flushes(sync=%llu event=%llu dtoh=%llu disp=%llu work=%llu)\n",
           (unsigned long long)f.stream->stats().submissions, (unsigned long long)f.stream->stats().dispatches,
           (unsigned long long)f.stream->stats().flushes_sync, (unsigned long long)f.stream->stats().flushes_event,
           (unsigned long long)f.stream->stats().flushes_dtoh, (unsigned long long)f.stream->stats().flushes_dispatch_threshold,
           (unsigned long long)f.stream->stats().flushes_work_threshold);
    printf("alloc: allocations=%llu reuses=%llu peak=%llu h2d=%llu d2h=%llu\n",
           (unsigned long long)f.alloc->stats().allocations, (unsigned long long)f.alloc->stats().reuses,
           (unsigned long long)f.alloc->stats().peak_bytes, (unsigned long long)f.alloc->stats().h2d_bytes,
           (unsigned long long)f.alloc->stats().d2h_bytes);
    f.dev.SavePipelineCache();
  } else {
    printf("SKIP Vulkan tests: no device (Tier C)\n");
  }
#endif
  printf(failures ? "FAILED (%d)\n" : "ALL PASS\n", failures);
  return failures ? 1 : 0;
}
