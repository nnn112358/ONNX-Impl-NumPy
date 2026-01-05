#ifndef ONNX_07_REDUCELOGSUM_HPP
#define ONNX_07_REDUCELOGSUM_HPP

#include <Eigen/Dense>
#include <cmath>

namespace onnx {

/**
 * ONNX ReduceLogSum operator
 *
 * 指定された軸に沿ってlog(sum(x))を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return log(sum(x))
 */
template<typename Derived>
auto reducelogsum(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // LogSum of all elements
        Scalar log_sum = std::log(data.sum());
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = log_sum;
            return result;
        } else {
            return log_sum;
        }
    } else if (axis == 0) {
        // LogSum along columns
        auto sums = data.colwise().sum();
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> result(1, data.cols());
        for (int j = 0; j < data.cols(); ++j) {
            result(0, j) = std::log(sums(j));
        }
        return result;
    } else {
        // LogSum along rows (axis == 1)
        auto sums = data.rowwise().sum();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(data.rows(), 1);
        for (int i = 0; i < data.rows(); ++i) {
            result(i, 0) = std::log(sums(i));
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCELOGSUM_HPP
