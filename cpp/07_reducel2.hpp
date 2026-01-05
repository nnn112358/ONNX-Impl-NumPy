#ifndef ONNX_07_REDUCEL2_HPP
#define ONNX_07_REDUCEL2_HPP

#include <Eigen/Dense>
#include <cmath>

namespace onnx {

/**
 * ONNX ReduceL2 operator
 *
 * 指定された軸に沿ってL2ノルムを計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return L2ノルム
 */
template<typename Derived>
auto reducel2(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // L2 norm of all elements
        Scalar norm = data.norm();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = norm;
            return result;
        } else {
            return norm;
        }
    } else if (axis == 0) {
        // L2 norm along columns
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> result(1, data.cols());
        for (int j = 0; j < data.cols(); ++j) {
            result(0, j) = data.col(j).norm();
        }
        return result;
    } else {
        // L2 norm along rows (axis == 1)
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(data.rows(), 1);
        for (int i = 0; i < data.rows(); ++i) {
            result(i, 0) = data.row(i).norm();
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEL2_HPP
