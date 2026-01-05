#ifndef ONNX_04_HARDSWISH_HPP
#define ONNX_04_HARDSWISH_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX HardSwish operator
 *
 * Hard Swish活性化関数。
 * f(x) = x * max(0, min(1, (x + 3) / 6))
 *
 * @param X 入力テンソル
 * @return Y: HardSwishを適用した結果
 */
template<typename Derived>
auto hardswish(const Eigen::MatrixBase<Derived>& X) {
    return (X.array() * ((X.array() + 3.0) / 6.0).max(0.0).min(1.0)).matrix();
}

} // namespace onnx

#endif // ONNX_04_HARDSWISH_HPP
