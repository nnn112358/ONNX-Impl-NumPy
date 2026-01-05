#ifndef ONNX_04_SIGMOID_HPP
#define ONNX_04_SIGMOID_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Sigmoid operator
 *
 * シグモイド活性化関数。
 * f(x) = 1 / (1 + exp(-x))
 *
 * @param X 入力テンソル
 * @return Y: Sigmoidを適用した結果（0から1の範囲）
 */
template<typename Derived>
auto sigmoid(const Eigen::MatrixBase<Derived>& X) {
    return (1.0 / (1.0 + (-X.array()).exp())).matrix();
}

} // namespace onnx

#endif // ONNX_04_SIGMOID_HPP
