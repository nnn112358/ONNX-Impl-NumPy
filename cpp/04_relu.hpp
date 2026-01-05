#ifndef ONNX_04_RELU_HPP
#define ONNX_04_RELU_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Relu operator
 *
 * ReLU（Rectified Linear Unit）活性化関数。
 * f(x) = max(0, x)
 *
 * @param X 入力テンソル
 * @return Y: ReLUを適用した結果
 */
template<typename Derived>
auto relu(const Eigen::MatrixBase<Derived>& X) {
    return X.array().max(0.0).matrix();
}

} // namespace onnx

#endif // ONNX_04_RELU_HPP
