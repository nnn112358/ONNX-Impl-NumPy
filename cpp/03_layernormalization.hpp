#ifndef ONNX_03_LAYERNORMALIZATION_HPP
#define ONNX_03_LAYERNORMALIZATION_HPP

#include <Eigen/Dense>
#include <cmath>

namespace onnx {

/**
 * ONNX LayerNormalization operator
 *
 * レイヤー正規化を行う。
 * Simplified implementation for last axis normalization.
 *
 * @param X 入力テンソル
 * @param scale スケールパラメータ (gamma)
 * @param bias バイアスパラメータ (beta)
 * @param epsilon 数値安定性のための小さな値
 * @return 正規化されたテンソル
 */
template<typename Derived1, typename Derived2, typename Derived3>
auto layernormalization(
    const Eigen::MatrixBase<Derived1>& X,
    const Eigen::MatrixBase<Derived2>& scale,
    const Eigen::MatrixBase<Derived3>& bias,
    double epsilon = 1e-5) {

    Eigen::MatrixXd result(X.rows(), X.cols());

    // Normalize each row independently
    for (int i = 0; i < X.rows(); ++i) {
        // Calculate mean
        double mean = X.row(i).mean();

        // Calculate variance
        double variance = 0.0;
        for (int j = 0; j < X.cols(); ++j) {
            double diff = X(i, j) - mean;
            variance += diff * diff;
        }
        variance /= X.cols();

        // Normalize and apply scale/bias
        double std_dev = std::sqrt(variance + epsilon);
        for (int j = 0; j < X.cols(); ++j) {
            double normalized = (X(i, j) - mean) / std_dev;
            result(i, j) = scale(j) * normalized + bias(j);
        }
    }

    return result;
}

} // namespace onnx

#endif // ONNX_03_LAYERNORMALIZATION_HPP
