#ifndef ONNX_07_REDUCELOGSUMEXP_HPP
#define ONNX_07_REDUCELOGSUMEXP_HPP

#include <Eigen/Dense>
#include <cmath>

namespace onnx {

/**
 * ONNX ReduceLogSumExp operator
 *
 * 指定された軸に沿ってlog(sum(exp(x)))を計算する。
 * 数値安定性のため、最大値を引いてから計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return log(sum(exp(x)))
 */
template<typename Derived>
auto reducelogsumexp(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // LogSumExp of all elements
        Scalar max_val = data.maxCoeff();
        Scalar lse = max_val + std::log((data.array() - max_val).exp().sum());
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = lse;
            return result;
        } else {
            return lse;
        }
    } else if (axis == 0) {
        // LogSumExp along columns
        Eigen::Matrix<Scalar, 1, Eigen::Dynamic> result(1, data.cols());
        for (int j = 0; j < data.cols(); ++j) {
            Scalar max_val = data.col(j).maxCoeff();
            result(0, j) = max_val + std::log((data.col(j).array() - max_val).exp().sum());
        }
        return result;
    } else {
        // LogSumExp along rows (axis == 1)
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> result(data.rows(), 1);
        for (int i = 0; i < data.rows(); ++i) {
            Scalar max_val = data.row(i).maxCoeff();
            result(i, 0) = max_val + std::log((data.row(i).array() - max_val).exp().sum());
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCELOGSUMEXP_HPP
