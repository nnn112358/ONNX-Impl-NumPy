#ifndef ONNX_04_SOFTMAX_HPP
#define ONNX_04_SOFTMAX_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Softmax operator
 *
 * Softmax関数を適用する。
 * 数値安定性のため、最大値を引いてから計算する。
 *
 * @param X 入力テンソル
 * @param axis Softmaxを適用する軸 (0: 列方向, 1: 行方向, デフォルト: 1)
 * @return Y: Softmaxを適用した結果（確率分布）
 */
template<typename Derived>
auto softmax(const Eigen::MatrixBase<Derived>& X, int axis = 1) {
    typedef typename Derived::Scalar Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> result = X;

    if (axis == 1) {
        // Row-wise softmax
        for (int i = 0; i < X.rows(); ++i) {
            Scalar max_val = X.row(i).maxCoeff();
            auto shifted = (X.row(i).array() - max_val).exp();
            Scalar sum = shifted.sum();
            result.row(i) = shifted / sum;
        }
    } else if (axis == 0) {
        // Column-wise softmax
        for (int j = 0; j < X.cols(); ++j) {
            Scalar max_val = X.col(j).maxCoeff();
            auto shifted = (X.col(j).array() - max_val).exp();
            Scalar sum = shifted.sum();
            result.col(j) = shifted / sum;
        }
    }

    return result;
}

} // namespace onnx

#endif // ONNX_04_SOFTMAX_HPP
