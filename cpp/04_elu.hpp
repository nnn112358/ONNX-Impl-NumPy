#ifndef ONNX_04_ELU_HPP
#define ONNX_04_ELU_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Elu operator
 *
 * ELU（Exponential Linear Unit）活性化関数。
 * f(x) = x if x >= 0 else alpha * (exp(x) - 1)
 *
 * @param X 入力テンソル
 * @param alpha 負の入力に対するスケール (デフォルト: 1.0)
 * @return Y: ELUを適用した結果
 */
template<typename Derived>
auto elu(const Eigen::MatrixBase<Derived>& X, double alpha = 1.0) {
    return (X.array() >= 0.0).select(X.array(), alpha * (X.array().exp() - 1.0)).matrix();
}

} // namespace onnx

#endif // ONNX_04_ELU_HPP
