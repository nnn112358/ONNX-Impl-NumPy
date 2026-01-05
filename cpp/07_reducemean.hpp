#ifndef ONNX_07_REDUCEMEAN_HPP
#define ONNX_07_REDUCEMEAN_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX ReduceMean operator
 *
 * 指定された軸に沿って平均を計算する。
 *
 * @param data 入力テンソル
 * @param axis 削減する軸 (0: 列方向, 1: 行方向, -1: 全要素, デフォルト: -1)
 * @param keepdims 次元を保持するか (デフォルト: true)
 * @return 平均値
 */
template<typename Derived>
auto reducemean(const Eigen::MatrixBase<Derived>& data, int axis = -1, bool keepdims = true) {
    typedef typename Derived::Scalar Scalar;

    if (axis == -1) {
        // Mean of all elements
        Scalar mean = data.mean();
        if (keepdims) {
            Eigen::Matrix<Scalar, 1, 1> result;
            result(0, 0) = mean;
            return result;
        } else {
            return mean;
        }
    } else if (axis == 0) {
        // Mean along columns
        auto mean = data.colwise().mean();
        if (keepdims) {
            return mean.eval();
        } else {
            return mean.eval();
        }
    } else {
        // Mean along rows (axis == 1)
        auto mean = data.rowwise().mean();
        if (keepdims) {
            return mean.eval();
        } else {
            return mean.eval();
        }
    }
}

} // namespace onnx

#endif // ONNX_07_REDUCEMEAN_HPP
