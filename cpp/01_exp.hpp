#ifndef ONNX_01_EXP_HPP
#define ONNX_01_EXP_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Exp operator
 *
 * テンソルの要素ごとに自然指数関数 e^x を計算する。
 *
 * @param X 入力テンソル
 * @return Y: e^X の結果
 */
template<typename Derived>
auto exp(const Eigen::MatrixBase<Derived>& X) {
    return X.array().exp().matrix();
}

} // namespace onnx

#endif // ONNX_01_EXP_HPP
