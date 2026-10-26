#pragma once
#include <string>
#include <utility>

namespace tfvk {

enum class Code {
  kOk, kInvalidArgument, kResourceExhausted, kUnavailable,
  kUnsupported, kInternal, kDeviceLost,
};

struct Status {
  Code code = Code::kOk;
  std::string message;
  bool ok() const { return code == Code::kOk; }
  static Status Ok() { return {}; }
  static Status Error(Code c, std::string m) { return {c, std::move(m)}; }
};

template <typename T>
struct Result {
  Status status;
  T value{};
  bool ok() const { return status.ok(); }
};

}  // namespace tfvk
