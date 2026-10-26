#pragma once
#include <cstdint>
#include <vector>

#include "status.h"

namespace tfvk {

enum class DType { kF32, kF16, kI32, kI64, kBool };

inline uint32_t DTypeSize(DType t) {
  switch (t) {
    case DType::kF32: case DType::kI32: return 4;
    case DType::kF16: return 2;
    case DType::kI64: return 8;
    case DType::kBool: return 1;
  }
  return 0;
}

// Tensor descriptor (spec §6.4): contiguous layout only for now.
struct TensorDesc {
  DType dtype = DType::kF32;
  std::vector<int64_t> dims;

  int64_t rank() const { return static_cast<int64_t>(dims.size()); }

  // Overflow-checked element count (spec §6.4/§12).
  Result<int64_t> numel() const {
    __int128 n = 1;
    for (int64_t d : dims) {
      if (d < 0) return {Status::Error(Code::kInvalidArgument, "negative dim"), 0};
      n *= d;
      if (n > INT64_MAX) return {Status::Error(Code::kInvalidArgument, "numel overflow"), 0};
    }
    return {Status::Ok(), static_cast<int64_t>(n)};
  }

  Result<uint64_t> byte_size() const {
    auto n = numel();
    if (!n.ok()) return {n.status, 0};
    __int128 b = static_cast<__int128>(n.value) * DTypeSize(dtype);
    if (b > static_cast<__int128>(UINT64_MAX))
      return {Status::Error(Code::kInvalidArgument, "byte size overflow"), 0};
    return {Status::Ok(), static_cast<uint64_t>(b)};
  }

  std::vector<int64_t> contiguous_strides() const {
    std::vector<int64_t> s(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i)
      s[i] = s[i + 1] * dims[i + 1];
    return s;
  }
};

// Validates offset+size against a physical buffer (spec §12).
inline Status CheckRange(uint64_t buffer_bytes, uint64_t offset, uint64_t size) {
  if (offset > buffer_bytes || size > buffer_bytes - offset)
    return Status::Error(Code::kInvalidArgument, "buffer range out of bounds");
  return Status::Ok();
}

}  // namespace tfvk
