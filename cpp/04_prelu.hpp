#ifndef ONNX_04_PRELU_HPP
#define ONNX_04_PRELU_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX PRelu operator
 *
 * PReLU（Parametric ReLU）活性化関数。
 * f(x) = x if x >= 0 else slope * x
 *
 * @param X 入力テンソル
 * @param slope 負の入力に対する傾きパラメータ
 * @return Y: PReL uを適用した結果
 */
template<typename Derived1, typename Derived2>
auto prelu(const Eigen::MatrixBase<Derived1>& X,
           const Eigen::MatrixBase<Derived2>& slope) {
    if (slope.size() == 1) {
        // Scalar slope
        return (X.array() >= 0.0).select(X.array(), slope(0) * X.array()).matrix();
    } else {
        // Element-wise or broadcasted slope
        return (X.array() >= 0.0).select(X.array(), slope.array() * X.array()).matrix();
    }
}

} // namespace onnx

#endif // ONNX_04_PRELU_HPP
