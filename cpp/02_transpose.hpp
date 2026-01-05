#ifndef ONNX_02_TRANSPOSE_HPP
#define ONNX_02_TRANSPOSE_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Transpose operator
 *
 * テンソルの次元を入れ替える。
 * 2D行列の場合は転置を行う。
 *
 * @param data 入力テンソル
 * @return transposed: 転置されたテンソル
 */
template<typename Derived>
auto transpose(const Eigen::MatrixBase<Derived>& data) {
    return data.transpose().eval();
}

} // namespace onnx

#endif // ONNX_02_TRANSPOSE_HPP
