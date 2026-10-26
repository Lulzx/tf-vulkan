#include "kernels.h"

#include <algorithm>
#include <cmath>

namespace tfvk {

void CpuMatMul(const float* a, const float* b, float* c, int64_t M, int64_t N, int64_t K,
               bool trans_a, bool trans_b) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0;
      for (int64_t k = 0; k < K; ++k) {
        float av = trans_a ? a[k * M + i] : a[i * K + k];
        float bv = trans_b ? b[j * K + k] : b[k * N + j];
        acc += av * bv;
      }
      c[i * N + j] = acc;
    }
}

void CpuMatMulF64(const double* a, const double* b, double* c, int64_t M, int64_t N, int64_t K) {
  for (int64_t i = 0; i < M * N; ++i) c[i] = 0;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < K; ++k)
      for (int64_t j = 0; j < N; ++j) c[i * N + j] += a[i * K + k] * b[k * N + j];
}

float CpuBinary(BinaryOp op, float x, float y) {
  switch (op) {
    case BinaryOp::kAdd: return x + y;
    case BinaryOp::kSub: return x - y;
    case BinaryOp::kMul: return x * y;
    case BinaryOp::kDiv: return x / y;
    case BinaryOp::kDivNoNan: return y == 0 ? 0.0f : x / y;
    case BinaryOp::kPow: return std::pow(x, y);
    case BinaryOp::kMaximum: return std::max(x, y);
    case BinaryOp::kSquaredDiff: { float d = x - y; return d * d; }
  }
  return x;
}

float CpuUnary(UnaryOp op, float x) {
  switch (op) {
    case UnaryOp::kIdentity: return x;
    case UnaryOp::kRelu: return std::max(x, 0.0f);
    case UnaryOp::kRelu6: return std::min(std::max(x, 0.0f), 6.0f);
    case UnaryOp::kSigmoid: return 1.0f / (1.0f + std::exp(-x));
    case UnaryOp::kTanh: return std::tanh(x);
    case UnaryOp::kExp: return std::exp(x);
    case UnaryOp::kLog: return std::log(x);
    case UnaryOp::kSqrt: return std::sqrt(x);
    case UnaryOp::kNeg: return -x;
    case UnaryOp::kAbs: return std::abs(x);
    case UnaryOp::kSquare: return x * x;
  }
  return x;
}

void CpuConv2D(const float* x, const float* f, float* y, int64_t B, int64_t H, int64_t W,
               int64_t Cin, int64_t Cout, int64_t Kh, int64_t Kw, int64_t sH, int64_t sW,
               int64_t padH, int64_t padW, int64_t outH, int64_t outW) {
  for (int64_t b = 0; b < B; ++b)
    for (int64_t oh = 0; oh < outH; ++oh)
      for (int64_t ow = 0; ow < outW; ++ow)
        for (int64_t co = 0; co < Cout; ++co) {
          float acc = 0;
          for (int64_t kh = 0; kh < Kh; ++kh) {
            int64_t ih = oh * sH + kh - padH;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < Kw; ++kw) {
              int64_t iw = ow * sW + kw - padW;
              if (iw < 0 || iw >= W) continue;
              for (int64_t ci = 0; ci < Cin; ++ci)
                acc += x[((b * H + ih) * W + iw) * Cin + ci] *
                       f[((kh * Kw + kw) * Cin + ci) * Cout + co];
            }
          }
          y[((b * outH + oh) * outW + ow) * Cout + co] = acc;
        }
}

void CpuSoftmaxRow(const float* in, float* out, int64_t cols) {
  float m = in[0];
  for (int64_t i = 1; i < cols; ++i) m = std::max(m, in[i]);
  float s = 0;
  for (int64_t i = 0; i < cols; ++i) { out[i] = std::exp(in[i] - m); s += out[i]; }
  for (int64_t i = 0; i < cols; ++i) out[i] /= s;
}

}  // namespace tfvk
