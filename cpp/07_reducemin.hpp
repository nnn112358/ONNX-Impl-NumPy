#ifndef ONNX_07_REDUCEMIN_HPP
#define ONNX_07_REDUCEMIN_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceMin operator
 *
 * 指定された軸に沿って最小値を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 最小値
 */
template<typename Derived>
auto reducemin(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Min of all elements
        Scalar min_val = data.minCoeff();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = min_val;
            return result;
        } else {
            return min_val;
        }
    } else if (axis == 0) {
        // Min along columns
        auto min_val = data.colwise().minCoeff();
        if (keepdims) {
            return min_val.eval();
        } else {
            return min_val.eval();
        }
    } else {
        // Min along rows (axis == 1)
        auto min_val = data.rowwise().minCoeff();
        if (keepdims) {
            return min_val.eval();
        } else {
            return min_val.eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEMIN_HPP
