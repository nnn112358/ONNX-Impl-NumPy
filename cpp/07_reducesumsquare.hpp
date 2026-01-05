#ifndef ONNX_07_REDUCESUMSQUARE_HPP
#define ONNX_07_REDUCESUMSQUARE_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceSumSquare operator
 *
 * 指定された軸に沿って二乗和を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 二乗和
 */
template<typename Derived>
auto reducesumsquare(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Sum of squares of all elements
        Scalar sum_sq = data.array().square().sum();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = sum_sq;
            return result;
        } else {
            return sum_sq;
        }
    } else if (axis == 0) {
        // Sum of squares along columns
        auto sum_sq = data.array().square().colwise().sum();
        if (keepdims) {
            return sum_sq.matrix().eval();
        } else {
            return sum_sq.matrix().eval();
        }
    } else {
        // Sum of squares along rows (axis == 1)
        auto sum_sq = data.array().square().rowwise().sum();
        if (keepdims) {
            return sum_sq.matrix().eval();
        } else {
            return sum_sq.matrix().eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCESUMSQUARE_HPP
