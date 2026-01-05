#ifndef ONNX_01_SQRT_HPP
#define ONNX_01_SQRT_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Sqrt operator
 *
 * テンソルの要素ごとに平方根を計算する。
 *
 * @param X 入力テンソル (非負の値)
 * @return Y: sqrt(X) の結果
 */
template<typename Derived>
auto sqrt(const Eigen::MatrixBase<Derived>& X) {
    return X.array().sqrt().matrix();
}

} // namespace onnx

#endif // ONNX_01_SQRT_HPP
