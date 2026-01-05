#ifndef ONNX_04_HARDSIGMOID_HPP
#define ONNX_04_HARDSIGMOID_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX HardSigmoid operator
 *
 * Hard Sigmoid活性化関数（区分線形近似）。
 * f(x) = max(0, min(1, alpha * x + beta))
 *
 * @param X 入力テンソル
 * @param alpha 傾き (デフォルト: 0.2)
 * @param beta オフセット (デフォルト: 0.5)
 * @return Y: HardSigmoidを適用した結果（0から1の範囲）
 */
template<typename Derived>
auto hardsigmoid(const Eigen::MatrixBase<Derived>& X,
                 double alpha = 0.2,
                 double beta = 0.5) {
    return (alpha * X.array() + beta).max(0.0).min(1.0).matrix();
}

} // namespace onnx

#endif // ONNX_04_HARDSIGMOID_HPP
