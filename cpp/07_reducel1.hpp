#ifndef ONNX_07_REDUCEL1_HPP
#define ONNX_07_REDUCEL1_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceL1 operator
 *
 * 指定された軸に沿ってL1ノルム（絶対値の和）を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return L1ノルム
 */
template<typename Derived>
auto reducel1(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // L1 norm of all elements
        Scalar norm = data.array().abs().sum();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = norm;
            return result;
        } else {
            return norm;
        }
    } else if (axis == 0) {
        // L1 norm along columns
        auto norm = data.array().abs().colwise().sum();
        if (keepdims) {
            return norm.matrix().eval();
        } else {
            return norm.matrix().eval();
        }
    } else {
        // L1 norm along rows (axis == 1)
        auto norm = data.array().abs().rowwise().sum();
        if (keepdims) {
            return norm.matrix().eval();
        } else {
            return norm.matrix().eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEL1_HPP
