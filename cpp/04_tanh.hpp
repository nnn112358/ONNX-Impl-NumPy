#ifndef ONNX_04_TANH_HPP
#define ONNX_04_TANH_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Tanh operator
 *
 * ハイパボリックタンジェント活性化関数。
 * f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * @param X 入力テンソル
 * @return Y: Tanhを適用した結果（-1から1の範囲）
 */
template<typename Derived>
auto tanh(const Eigen::MatrixBase<Derived>& X) {
    return X.array().tanh().matrix();
}

} // namespace onnx

#endif // ONNX_04_TANH_HPP
