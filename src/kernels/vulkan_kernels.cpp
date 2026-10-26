#include "kernels.h"

#ifdef TFVK_HAS_VULKAN
#include "shaders_spv.h"

namespace tfvk {

Status VulkanKernels::MakePipeline(const uint32_t* spv, uint32_t words, uint32_t bindings,
                                   uint32_t push_bytes, VkPipeline* out, VkPipelineLayout* layout,
                                   VkDescriptorSetLayout* dsl) {
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = words * 4;
  smci.pCode = spv;
  VkShaderModule mod = VK_NULL_HANDLE;
  if (vkCreateShaderModule(dev_->device(), &smci, nullptr, &mod) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkCreateShaderModule failed");
  std::vector<VkDescriptorSetLayoutBinding> binds(bindings);
  for (uint32_t i = 0; i < bindings; ++i) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dci.bindingCount = bindings;
  dci.pBindings = binds.data();
  if (vkCreateDescriptorSetLayout(dev_->device(), &dci, nullptr, dsl) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkCreateDescriptorSetLayout failed");
  VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
  VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  lci.setLayoutCount = 1;
  lci.pSetLayouts = dsl;
  lci.pushConstantRangeCount = 1;
  lci.pPushConstantRanges = &range;
  if (vkCreatePipelineLayout(dev_->device(), &lci, nullptr, layout) != VK_SUCCESS)
    return Status::Error(Code::kInternal, "vkCreatePipelineLayout failed");
  VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = mod;
  pci.stage.pName = "main";
  pci.layout = *layout;
  VkResult r = vkCreateComputePipelines(dev_->device(), dev_->pipeline_cache(), 1, &pci, nullptr, out);
  vkDestroyShaderModule(dev_->device(), mod, nullptr);
  if (r != VK_SUCCESS) return Status::Error(Code::kInternal, "vkCreateComputePipelines failed");
  return Status::Ok();
}

Status VulkanKernels::Init() {
  if (dev_->info().max_push_constants < 32)
    return Status::Error(Code::kUnsupported, "maxPushConstantsSize < 32 bytes");
  Status s;
  if (!(s = MakePipeline(kMatmulSpv, kMatmulSpv_words, 3, 32, &matmul_.p, &matmul_.l, &matmul_.d)).ok()) return s;
  if (!(s = MakePipeline(kMatmulTiledSpv, kMatmulTiledSpv_words, 3, 32, &matmul_tiled_.p, &matmul_tiled_.l, &matmul_tiled_.d)).ok()) return s;
  if (!(s = MakePipeline(kBinarySpv, kBinarySpv_words, 3, 64, &binary_.p, &binary_.l, &binary_.d)).ok()) return s;
  if (!(s = MakePipeline(kUnarySpv, kUnarySpv_words, 2, 32, &unary_.p, &unary_.l, &unary_.d)).ok()) return s;
  if (!(s = MakePipeline(kBiasAddSpv, kBiasAddSpv_words, 3, 32, &bias_add_.p, &bias_add_.l, &bias_add_.d)).ok()) return s;
  if (!(s = MakePipeline(kReduceSpv, kReduceSpv_words, 2, 32, &reduce_.p, &reduce_.l, &reduce_.d)).ok()) return s;
  if (!(s = MakePipeline(kSoftmaxSpv, kSoftmaxSpv_words, 2, 32, &softmax_.p, &softmax_.l, &softmax_.d)).ok()) return s;
  if (!(s = MakePipeline(kGradSpv, kGradSpv_words, 3, 32, &grad_.p, &grad_.l, &grad_.d)).ok()) return s;
  if (!(s = MakePipeline(kConv2dSpv, kConv2dSpv_words, 3, 56, &conv2d_.p, &conv2d_.l, &conv2d_.d)).ok()) return s;
  if (!(s = MakePipeline(kPoolSpv, kPoolSpv_words, 2, 52, &pool_.p, &pool_.l, &pool_.d)).ok()) return s;
  if (!(s = MakePipeline(kDepthwiseSpv, kDepthwiseSpv_words, 3, 56, &depthwise_.p, &depthwise_.l, &depthwise_.d)).ok()) return s;
  if (!(s = MakePipeline(kConvFilterGradSpv, kConvFilterGradSpv_words, 3, 56, &conv_filter_grad_.p, &conv_filter_grad_.l, &conv_filter_grad_.d)).ok()) return s;
  if (!(s = MakePipeline(kConvInputGradSpv, kConvInputGradSpv_words, 3, 56, &conv_input_grad_.p, &conv_input_grad_.l, &conv_input_grad_.d)).ok()) return s;
  if (dev_->info().shader_float16) {
    if (!(s = MakePipeline(kMatmulF16Spv, kMatmulF16Spv_words, 3, 32, &matmul_f16_.p, &matmul_f16_.l, &matmul_f16_.d)).ok()) return s;
    if (!(s = MakePipeline(kUnaryF16Spv, kUnaryF16Spv_words, 2, 32, &unary_f16_.p, &unary_f16_.l, &unary_f16_.d)).ok()) return s;
    if (!(s = MakePipeline(kBinaryF16Spv, kBinaryF16Spv_words, 3, 32, &binary_f16_.p, &binary_f16_.l, &binary_f16_.d)).ok()) return s;
    if (!(s = MakePipeline(kBiasAddF16Spv, kBiasAddF16Spv_words, 3, 32, &bias_add_f16_.p, &bias_add_f16_.l, &bias_add_f16_.d)).ok()) return s;
    if (!(s = MakePipeline(kCastToF16Spv, kCastToF16Spv_words, 2, 32, &cast_to_f16_.p, &cast_to_f16_.l, &cast_to_f16_.d)).ok()) return s;
    if (!(s = MakePipeline(kCastFromF16Spv, kCastFromF16Spv_words, 2, 32, &cast_from_f16_.p, &cast_from_f16_.l, &cast_from_f16_.d)).ok()) return s;
  }
  ready_ = true;
  return Status::Ok();
}

Status VulkanKernels::Dispatch(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSetLayout dsl,
                               std::vector<VkBuffer> buffers, const void* push, uint32_t push_bytes,
                               uint32_t gx, uint32_t gy) {
  if (!ready_) return Status::Error(Code::kUnavailable, "kernels not initialized");
  auto ps = stream_->PrepareBatch();
  if (!ps.ok()) return ps;
  VkDescriptorPool pool = stream_->descriptor_pool();
  if (pool == VK_NULL_HANDLE) return Status::Error(Code::kInternal, "no descriptor pool for batch");
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = pool;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &dsl;
  VkDescriptorSet set = VK_NULL_HANDLE;
  if (vkAllocateDescriptorSets(dev_->device(), &ai, &set) != VK_SUCCESS)
    return Status::Error(Code::kResourceExhausted, "descriptor pool exhausted in batch");
  std::vector<VkDescriptorBufferInfo> infos(buffers.size());
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = set;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  uint64_t wg = static_cast<uint64_t>(gx) * gy;
  return stream_->Record([&](VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cmd, gx, gy, 1);
    // Same-batch ordering for chained ops (§6.3).
    VkMemoryBarrier bar{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    bar.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &bar, 0, nullptr, 0, nullptr);
  }, wg);
}

struct MatmulPC { uint32_t M, N, K, transA, transB, aOff, bOff, cOff; };
Status VulkanKernels::MatMul(Allocation* a, Allocation* b, Allocation* c, uint32_t M, uint32_t N,
                             uint32_t K, bool ta, bool tb, uint32_t ao, uint32_t bo, uint32_t co) {
  MatmulPC pc{M, N, K, ta ? 1u : 0u, tb ? 1u : 0u, ao, bo, co};
  bool tiled = !ta && !tb;
  Pipe& p = tiled ? matmul_tiled_ : matmul_;
  return Dispatch(p.p, p.l, p.d, {a->buffer, b->buffer, c->buffer}, &pc, sizeof(pc),
                  (N + 15) / 16, (M + 15) / 16);
}

struct BinaryPC {
  uint32_t numel, op, rank, pad;
  uint32_t outDims[4], aStrides[4], bStrides[4];
};
static void FillStrides(const std::vector<int64_t>& out, const std::vector<int64_t>& in,
                        uint32_t* strides) {
  int rank = (int)out.size();
  int shift = rank - (int)in.size();
  std::vector<int64_t> cont(in.size(), 1);
  for (int i = (int)in.size() - 2; i >= 0; --i) cont[i] = cont[i + 1] * in[i + 1];
  for (int d = 0; d < 4; ++d) strides[d] = 0;
  for (int d = 0; d < rank; ++d) {
    int id = d - shift;
    strides[d] = (id < 0 || in[id] == 1) ? 0 : (uint32_t)cont[id];
  }
}
Status VulkanKernels::BinaryBroadcast(BinaryOp op, Allocation* a, Allocation* b, Allocation* c,
                                      const std::vector<int64_t>& out_dims,
                                      const std::vector<int64_t>& a_dims,
                                      const std::vector<int64_t>& b_dims) {
  if (out_dims.size() > 4)
    return Status::Error(Code::kUnsupported, "binary broadcast rank > 4");
  BinaryPC pc{};
  pc.op = static_cast<uint32_t>(op);
  pc.rank = (uint32_t)out_dims.size();
  uint32_t numel = 1;
  for (size_t d = 0; d < 4; ++d) {
    pc.outDims[d] = d < out_dims.size() ? (uint32_t)out_dims[d] : 1;
    if (d < out_dims.size()) numel *= pc.outDims[d];
  }
  if (out_dims.empty()) numel = 1;
  pc.numel = numel;
  FillStrides(out_dims, a_dims, pc.aStrides);
  FillStrides(out_dims, b_dims, pc.bStrides);
  return Dispatch(binary_.p, binary_.l, binary_.d, {a->buffer, b->buffer, c->buffer}, &pc,
                  sizeof(pc), (numel + 255) / 256, 1);
}
Status VulkanKernels::Binary(BinaryOp op, Allocation* a, Allocation* b, Allocation* c,
                             uint32_t numel, bool b_broadcast) {
  std::vector<int64_t> out{(int64_t)numel};
  std::vector<int64_t> bd = b_broadcast ? std::vector<int64_t>{1} : out;
  return BinaryBroadcast(op, a, b, c, out, out, bd);
}

struct UnaryPC { uint32_t numel, op, aOff, cOff, p0, p1, p2, p3; };
Status VulkanKernels::Unary(UnaryOp op, Allocation* a, Allocation* c, uint32_t numel) {
  UnaryPC pc{numel, static_cast<uint32_t>(op), 0, 0, 0, 0, 0, 0};
  return Dispatch(unary_.p, unary_.l, unary_.d, {a->buffer, c->buffer}, &pc, sizeof(pc),
                  (numel + 255) / 256, 1);
}

struct BiasPC { uint32_t numel, channels, xOff, bOff, cOff, p0, p1, p2; };
Status VulkanKernels::BiasAdd(Allocation* x, Allocation* bias, Allocation* c, uint32_t numel, uint32_t channels) {
  BiasPC pc{numel, channels, 0, 0, 0, 0, 0, 0};
  return Dispatch(bias_add_.p, bias_add_.l, bias_add_.d, {x->buffer, bias->buffer, c->buffer}, &pc,
                  sizeof(pc), (numel + 255) / 256, 1);
}

struct ReducePC { uint32_t outer, axisDim, inner, op, aOff, cOff, p0, p1; };
Status VulkanKernels::Reduce(ReduceOp op, Allocation* a, Allocation* c, uint32_t outer,
                             uint32_t axis_dim, uint32_t inner) {
  ReducePC pc{outer, axis_dim, inner, static_cast<uint32_t>(op), 0, 0, 0, 0};
  return Dispatch(reduce_.p, reduce_.l, reduce_.d, {a->buffer, c->buffer}, &pc, sizeof(pc),
                  (outer * inner + 255) / 256, 1);
}

struct SoftmaxPC { uint32_t rows, cols, aOff, cOff, p0, p1, p2, p3; };
Status VulkanKernels::Softmax(Allocation* a, Allocation* c, uint32_t rows, uint32_t cols) {
  SoftmaxPC pc{rows, cols, 0, 0, 0, 0, 0, 0};
  return Dispatch(softmax_.p, softmax_.l, softmax_.d, {a->buffer, c->buffer}, &pc, sizeof(pc),
                  rows, 1);
}

Status VulkanKernels::MatMulF16(Allocation* a, Allocation* b, Allocation* c, uint32_t M, uint32_t N,
                                uint32_t K, bool ta, bool tb, uint32_t ao, uint32_t bo, uint32_t co) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  MatmulPC pc{M, N, K, ta ? 1u : 0u, tb ? 1u : 0u, ao, bo, co};
  return Dispatch(matmul_f16_.p, matmul_f16_.l, matmul_f16_.d, {a->buffer, b->buffer, c->buffer},
                  &pc, sizeof(pc), (N + 15) / 16, (M + 15) / 16);
}
Status VulkanKernels::UnaryF16(UnaryOp op, Allocation* a, Allocation* c, uint32_t numel) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  UnaryPC pc{numel, static_cast<uint32_t>(op), 0, 0, 0, 0, 0, 0};
  return Dispatch(unary_f16_.p, unary_f16_.l, unary_f16_.d, {a->buffer, c->buffer}, &pc, sizeof(pc),
                  (numel + 255) / 256, 1);
}
Status VulkanKernels::BinaryF16(BinaryOp op, Allocation* a, Allocation* b, Allocation* c,
                                uint32_t numel, bool b_broadcast) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  struct BPC { uint32_t numel, op, bBroadcast, aOff, bOff, cOff, p0, p1; };
  BPC pc{numel, static_cast<uint32_t>(op), b_broadcast ? 1u : 0u, 0, 0, 0, 0, 0};
  return Dispatch(binary_f16_.p, binary_f16_.l, binary_f16_.d, {a->buffer, b->buffer, c->buffer},
                  &pc, sizeof(pc), (numel + 255) / 256, 1);
}
Status VulkanKernels::BiasAddF16(Allocation* x, Allocation* bias, Allocation* c, uint32_t numel,
                                 uint32_t channels) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  BiasPC pc{numel, channels, 0, 0, 0, 0, 0, 0};
  return Dispatch(bias_add_f16_.p, bias_add_f16_.l, bias_add_f16_.d,
                  {x->buffer, bias->buffer, c->buffer}, &pc, sizeof(pc), (numel + 255) / 256, 1);
}
Status VulkanKernels::CastToF16(Allocation* a, Allocation* c, uint32_t numel) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  UnaryPC pc{numel, 0, 0, 0, 0, 0, 0, 0};
  return Dispatch(cast_to_f16_.p, cast_to_f16_.l, cast_to_f16_.d, {a->buffer, c->buffer}, &pc,
                  sizeof(pc), (numel + 255) / 256, 1);
}
Status VulkanKernels::CastFromF16(Allocation* a, Allocation* c, uint32_t numel) {
  if (!dev_->info().shader_float16)
    return Status::Error(Code::kUnsupported, "FP16 path gated off: shaderFloat16 unavailable (§6.4)");
  UnaryPC pc{numel, 0, 0, 0, 0, 0, 0, 0};
  return Dispatch(cast_from_f16_.p, cast_from_f16_.l, cast_from_f16_.d, {a->buffer, c->buffer}, &pc,
                  sizeof(pc), (numel + 255) / 256, 1);
}

struct ConvPC {
  uint32_t B, H, W, Cin, Cout, Kh, Kw, sH, sW, padH, padW, outH, outW, pad2;
};
Status VulkanKernels::Conv2D(Allocation* x, Allocation* f, Allocation* y, uint32_t B, uint32_t H,
                             uint32_t W, uint32_t Cin, uint32_t Cout, uint32_t Kh, uint32_t Kw,
                             uint32_t sH, uint32_t sW, uint32_t padH, uint32_t padW,
                             uint32_t outH, uint32_t outW) {
  ConvPC pc{B, H, W, Cin, Cout, Kh, Kw, sH, sW, padH, padW, outH, outW, 0};
  return Dispatch(conv2d_.p, conv2d_.l, conv2d_.d, {x->buffer, f->buffer, y->buffer}, &pc,
                  sizeof(pc), (outW * Cout + 15) / 16, (B * outH + 15) / 16);
}

using ConvGeom = ConvPC;
Status VulkanKernels::Pool(bool avg, Allocation* x, Allocation* y, uint32_t B, uint32_t H,
                           uint32_t W, uint32_t C, uint32_t Kh, uint32_t Kw, uint32_t sH,
                           uint32_t sW, uint32_t padH, uint32_t padW, uint32_t outH, uint32_t outW) {
  struct PoolPC { uint32_t B,H,W,C,Kh,Kw,sH,sW,padH,padW,outH,outW,mode; };
  PoolPC pc{B,H,W,C,Kh,Kw,sH,sW,padH,padW,outH,outW, avg?1u:0u};
  return Dispatch(pool_.p, pool_.l, pool_.d, {x->buffer, y->buffer}, &pc, sizeof(pc),
                  (outW*C+15)/16, (B*outH+15)/16);
}
Status VulkanKernels::DepthwiseConv(Allocation* x, Allocation* f, Allocation* y, uint32_t B,
                                    uint32_t H, uint32_t W, uint32_t Cin, uint32_t mult,
                                    uint32_t Kh, uint32_t Kw, uint32_t sH, uint32_t sW,
                                    uint32_t padH, uint32_t padW, uint32_t outH, uint32_t outW) {
  ConvPC pc{B,H,W,Cin,mult,Kh,Kw,sH,sW,padH,padW,outH,outW,0};
  uint32_t cout = Cin*mult;
  return Dispatch(depthwise_.p, depthwise_.l, depthwise_.d, {x->buffer, f->buffer, y->buffer}, &pc,
                  sizeof(pc), (outW*cout+15)/16, (B*outH+15)/16);
}
Status VulkanKernels::ConvFilterGrad(Allocation* x, Allocation* g, Allocation* y, uint32_t B,
                                     uint32_t H, uint32_t W, uint32_t Cin, uint32_t Cout,
                                     uint32_t Kh, uint32_t Kw, uint32_t sH, uint32_t sW,
                                     uint32_t padH, uint32_t padW, uint32_t outH, uint32_t outW) {
  ConvPC pc{B,H,W,Cin,Cout,Kh,Kw,sH,sW,padH,padW,outH,outW,0};
  return Dispatch(conv_filter_grad_.p, conv_filter_grad_.l, conv_filter_grad_.d,
                  {x->buffer, g->buffer, y->buffer}, &pc, sizeof(pc), (Kh*Kw*Cin+15)/16, Cout);
}
Status VulkanKernels::ConvInputGrad(Allocation* g, Allocation* f, Allocation* y, uint32_t B,
                                    uint32_t H, uint32_t W, uint32_t Cin, uint32_t Cout,
                                    uint32_t Kh, uint32_t Kw, uint32_t sH, uint32_t sW,
                                    uint32_t padH, uint32_t padW, uint32_t outH, uint32_t outW) {
  ConvPC pc{B,H,W,Cin,Cout,Kh,Kw,sH,sW,padH,padW,outH,outW,0};
  return Dispatch(conv_input_grad_.p, conv_input_grad_.l, conv_input_grad_.d,
                  {g->buffer, f->buffer, y->buffer}, &pc, sizeof(pc), (W*Cin+15)/16, (B*H+15)/16);
}

struct GradPC { uint32_t numel, op, p0, p1, p2, p3, p4, p5; };
Status VulkanKernels::ReluGrad(Allocation* grad, Allocation* x, Allocation* c, uint32_t numel) {
  GradPC pc{numel, 0, 0, 0, 0, 0, 0, 0};
  return Dispatch(grad_.p, grad_.l, grad_.d, {grad->buffer, x->buffer, c->buffer}, &pc, sizeof(pc),
                  (numel + 255) / 256, 1);
}

Status VulkanKernels::Copy(Allocation* src, Allocation* dst, uint64_t size) {
  return stream_->Record([&](VkCommandBuffer cmd) {
    // Compute->transfer and transfer->compute ordering (§6.3): a plain
    // vkCmdCopyBuffer otherwise races the surrounding dispatches.
    VkMemoryBarrier pre{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0, nullptr, 0, nullptr);
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, src->buffer, dst->buffer, 1, &region);
    VkMemoryBarrier post{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &post, 0, nullptr, 0, nullptr);
  }, 1);
}

}  // namespace tfvk
#endif
