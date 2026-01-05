#ifndef ONNX_01_LOG_HPP
#define ONNX_01_LOG_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Log operator
 *
 * テンソルの要素ごとに自然対数を計算する。
 *
 * @param X 入力テンソル (正の値)
 * @return Y: log(X) の結果
 */
template<typename Derived>
auto log(const Eigen::MatrixBase<Derived>& X) {
    return X.array().log().matrix();
}

} // namespace onnx

#endif // ONNX_01_LOG_HPP
