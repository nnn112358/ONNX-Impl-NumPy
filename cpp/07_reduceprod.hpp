#ifndef ONNX_07_REDUCEPROD_HPP
#define ONNX_07_REDUCEPROD_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceProd operator
 *
 * 指定された軸に沿って積を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 積
 */
template<typename Derived>
auto reduceprod(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Product of all elements
        Scalar prod = data.prod();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = prod;
            return result;
        } else {
            return prod;
        }
    } else if (axis == 0) {
        // Product along columns
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> result(1, data.cols());
        for (int j = 0; j < data.cols(); ++j) {
            result(0, j) = data.col(j).prod();
        }
        return result;
    } else {
        // Product along rows (axis == 1)
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(data.rows(), 1);
        for (int i = 0; i < data.rows(); ++i) {
            result(i, 0) = data.row(i).prod();
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEPROD_HPP
