#ifndef ONNX_07_REDUCESUM_HPP
#define ONNX_07_REDUCESUM_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceSum operator
 *
 * 指定された軸に沿って合計を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 合計値
 */
template<typename Derived>
auto reducesum(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Sum all elements
        Scalar sum = data.sum();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = sum;
            return result;
        } else {
            return sum;
        }
    } else if (axis == 0) {
        // Sum along columns
        auto sum = data.colwise().sum();
        if (keepdims) {
            return sum.eval();
        } else {
            return sum.eval();
        }
    } else {
        // Sum along rows (axis == 1)
        auto sum = data.rowwise().sum();
        if (keepdims) {
            return sum.eval();
        } else {
            return sum.eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCESUM_HPP
