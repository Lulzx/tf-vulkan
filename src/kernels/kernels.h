#pragma once
#include <cstdint>
#include <vector>

#include "../runtime/allocator.h"
#include "../runtime/stream.h"
#include "../runtime/tensor.h"

namespace tfvk {

// ---- CPU reference (correctness oracle; float64 MatMul available in tests) ----
void CpuMatMul(const float* a, const float* b, float* c, int64_t M, int64_t N, int64_t K,
               bool trans_a = false, bool trans_b = false);
void CpuMatMulF64(const double* a, const double* b, double* c, int64_t M, int64_t N, int64_t K);

enum class BinaryOp : uint32_t { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kDivNoNan = 4, kPow = 5, kMaximum = 6, kSquaredDiff = 7 };
enum class UnaryOp : uint32_t { kIdentity = 0, kRelu = 1, kRelu6 = 2, kSigmoid = 3, kTanh = 4, kExp = 5, kLog = 6, kSqrt = 7, kNeg = 8, kAbs = 9, kSquare = 11 };
enum class ReduceOp : uint32_t { kSum = 0, kMean = 1, kMax = 2 };

float CpuBinary(BinaryOp op, float x, float y);
float CpuUnary(UnaryOp op, float x);
void CpuSoftmaxRow(const float* in, float* out, int64_t cols);
void CpuConv2D(const float* x, const float* f, float* y, int64_t B, int64_t H, int64_t W,
               int64_t Cin, int64_t Cout, int64_t Kh, int64_t Kw, int64_t sH, int64_t sW,
               int64_t padH, int64_t padW, int64_t outH, int64_t outW);

#ifdef TFVK_HAS_VULKAN
// Vulkan eager kernels (spec §7 K0/K1). All record into the deferred stream;
// buffers are FP32 storage buffers, offsets are in elements.
class VulkanKernels {
 public:
  VulkanKernels(VulkanDevice* dev, Stream* stream) : dev_(dev), stream_(stream) {}
  Status Init();
  Status MatMul(Allocation* a, Allocation* b, Allocation* c, uint32_t M, uint32_t N, uint32_t K,
                bool trans_a = false, bool trans_b = false,
                uint32_t a_off = 0, uint32_t b_off = 0, uint32_t c_off = 0);
  Status Binary(BinaryOp op, Allocation* a, Allocation* b, Allocation* c, uint32_t numel, bool b_broadcast);
  // Broadcasting variant: dims are logical shapes; rank <= 4.
  Status BinaryBroadcast(BinaryOp op, Allocation* a, Allocation* b, Allocation* c,
                         const std::vector<int64_t>& out_dims, const std::vector<int64_t>& a_dims,
                         const std::vector<int64_t>& b_dims);
  Status Unary(UnaryOp op, Allocation* a, Allocation* c, uint32_t numel);
  Status BiasAdd(Allocation* x, Allocation* bias, Allocation* c, uint32_t numel, uint32_t channels);
  Status Reduce(ReduceOp op, Allocation* a, Allocation* c, uint32_t outer, uint32_t axis_dim, uint32_t inner);
  Status Softmax(Allocation* a, Allocation* c, uint32_t rows, uint32_t cols);
  Status ReluGrad(Allocation* grad, Allocation* x, Allocation* c, uint32_t numel);
  // Opt-in FP16 path (K1.5): f16 storage, FP32 accumulation/math.
  Status MatMulF16(Allocation* a, Allocation* b, Allocation* c, uint32_t M, uint32_t N, uint32_t K,
                   bool trans_a = false, bool trans_b = false, uint32_t a_off = 0,
                   uint32_t b_off = 0, uint32_t c_off = 0);
  Status UnaryF16(UnaryOp op, Allocation* a, Allocation* c, uint32_t numel);
  Status BinaryF16(BinaryOp op, Allocation* a, Allocation* b, Allocation* c, uint32_t numel, bool b_broadcast);
  Status BiasAddF16(Allocation* x, Allocation* bias, Allocation* c, uint32_t numel, uint32_t channels);
  Status CastToF16(Allocation* a, Allocation* c, uint32_t numel);
  Status CastFromF16(Allocation* a, Allocation* c, uint32_t numel);
  Status Conv2D(Allocation* x, Allocation* f, Allocation* y, uint32_t B, uint32_t H, uint32_t W,
                uint32_t Cin, uint32_t Cout, uint32_t Kh, uint32_t Kw, uint32_t sH, uint32_t sW,
                uint32_t padH, uint32_t padW, uint32_t outH, uint32_t outW);
  Status Pool(bool avg, Allocation* x, Allocation* y, uint32_t B, uint32_t H, uint32_t W,
              uint32_t C, uint32_t Kh, uint32_t Kw, uint32_t sH, uint32_t sW, uint32_t padH,
              uint32_t padW, uint32_t outH, uint32_t outW);
  Status DepthwiseConv(Allocation* x, Allocation* f, Allocation* y, uint32_t B, uint32_t H,
                       uint32_t W, uint32_t Cin, uint32_t mult, uint32_t Kh, uint32_t Kw,
                       uint32_t sH, uint32_t sW, uint32_t padH, uint32_t padW, uint32_t outH,
                       uint32_t outW);
  Status ConvFilterGrad(Allocation* x, Allocation* g, Allocation* y, uint32_t B, uint32_t H,
                        uint32_t W, uint32_t Cin, uint32_t Cout, uint32_t Kh, uint32_t Kw,
                        uint32_t sH, uint32_t sW, uint32_t padH, uint32_t padW, uint32_t outH,
                        uint32_t outW);
  Status ConvInputGrad(Allocation* g, Allocation* f, Allocation* y, uint32_t B, uint32_t H,
                       uint32_t W, uint32_t Cin, uint32_t Cout, uint32_t Kh, uint32_t Kw,
                       uint32_t sH, uint32_t sW, uint32_t padH, uint32_t padW, uint32_t outH,
                       uint32_t outW);
  Status Copy(Allocation* src, Allocation* dst, uint64_t size);  // device-to-device

 private:
  Status MakePipeline(const uint32_t* spv, uint32_t words, uint32_t bindings,
                      uint32_t push_bytes, VkPipeline* out, VkPipelineLayout* layout,
                      VkDescriptorSetLayout* dsl);
  Status Dispatch(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSetLayout dsl,
                  std::vector<VkBuffer> buffers, const void* push, uint32_t push_bytes,
                  uint32_t gx, uint32_t gy);
  VulkanDevice* dev_;
  Stream* stream_;
  struct Pipe { VkPipeline p = VK_NULL_HANDLE; VkPipelineLayout l = VK_NULL_HANDLE; VkDescriptorSetLayout d = VK_NULL_HANDLE; };
  Pipe matmul_, matmul_tiled_, binary_, unary_, bias_add_, reduce_, softmax_, grad_, conv2d_, pool_, depthwise_, conv_filter_grad_, conv_input_grad_, matmul_f16_, unary_f16_, binary_f16_, bias_add_f16_, cast_to_f16_, cast_from_f16_;
  bool ready_ = false;
};
#endif

}  // namespace tfvk
