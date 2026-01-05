#ifndef ONNX_07_REDUCEMAX_HPP
#define ONNX_07_REDUCEMAX_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceMax operator
 *
 * 指定された軸に沿って最大値を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 最大値
 */
template<typename Derived>
auto reducemax(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Max of all elements
        Scalar max_val = data.maxCoeff();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = max_val;
            return result;
        } else {
            return max_val;
        }
    } else if (axis == 0) {
        // Max along columns
        auto max_val = data.colwise().maxCoeff();
        if (keepdims) {
            return max_val.eval();
        } else {
            return max_val.eval();
        }
    } else {
        // Max along rows (axis == 1)
        auto max_val = data.rowwise().maxCoeff();
        if (keepdims) {
            return max_val.eval();
        } else {
            return max_val.eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEMAX_HPP
