#ifndef ONNX_03_GLOBALAVERAGEPOOL_HPP
#define ONNX_03_GLOBALAVERAGEPOOL_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX GlobalAveragePool operator
 *
 * 各チャネルの空間次元全体にわたる平均値プーリングを行う。
 * Simplified 2D implementation for (1, C, H, W) input.
 *
 * @param X 入力テンソル (C x (H*W))
 * @param C チャネル数
 * @param H 入力高さ
 * @param W 入力幅
 * @return 出力テンソル (C x 1) - each channel's global average
 */
template<typename Derived>
auto globalaveragepool(const Eigen::MatrixBase<Derived>& X, int C, int H, int W) {
    Eigen::MatrixXd result(C, 1);

    int spatial_size = H * W;

    for (int c = 0; c < C; ++c) {
        double sum = 0.0;
        for (int i = 0; i < spatial_size; ++i) {
            sum += X(c, i);
        }
        result(c, 0) = sum / spatial_size;
    }

    return result;
}

} // namespace onnx

#endif // ONNX_03_GLOBALAVERAGEPOOL_HPP
