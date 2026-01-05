#ifndef ONNX_04_SWISH_HPP
#define ONNX_04_SWISH_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Swish operator
 *
 * Swish活性化関数（SiLUとも呼ばれる）。
 * f(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * @param X 入力テンソル
 * @return Y: Swishを適用した結果
 */
template<typename Derived>
auto swish(const Eigen::MatrixBase<Derived>& X) {
    return (X.array() / (1.0 + (-X.array()).exp())).matrix();
}

} // namespace onnx

#endif // ONNX_04_SWISH_HPP
