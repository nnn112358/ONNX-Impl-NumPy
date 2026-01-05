#ifndef ONNX_04_LEAKYRELU_HPP
#define ONNX_04_LEAKYRELU_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX LeakyRelu operator
 *
 * Leaky ReLU活性化関数。
 * f(x) = x if x >= 0 else alpha * x
 *
 * @param X 入力テンソル
 * @param alpha 負の入力に対する傾き (デフォルト: 0.01)
 * @return Y: LeakyReLUを適用した結果
 */
template<typename Derived>
auto leakyrelu(const Eigen::MatrixBase<Derived>& X, double alpha = 0.01) {
    return (X.array() >= 0.0).select(X.array(), alpha * X.array()).matrix();
}

} // namespace onnx

#endif // ONNX_04_LEAKYRELU_HPP
