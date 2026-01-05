#ifndef ONNX_01_NEG_HPP
#define ONNX_01_NEG_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Neg operator
 *
 * テンソルの要素ごとに符号を反転する。
 *
 * @param X 入力テンソル
 * @return Y: -X の結果
 */
template<typename Derived>
auto neg(const Eigen::MatrixBase<Derived>& X) {
    return (-X.array()).matrix();
}

} // namespace onnx

#endif // ONNX_01_NEG_HPP
