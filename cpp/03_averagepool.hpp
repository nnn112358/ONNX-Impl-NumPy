#ifndef ONNX_03_AVERAGEPOOL_HPP
#define ONNX_03_AVERAGEPOOL_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX AveragePool operator
 *
 * 平均値プーリング演算を行う。
 * Simplified 2D implementation for (1, C, H, W) input.
 *
 * @param X 入力テンソル (flattened from N=1, C, H, W)
 * @param C チャネル数
 * @param H 入力高さ
 * @param W 入力幅
 * @param kernel_h カーネル高さ
 * @param kernel_w カーネル幅
 * @param stride_h ストライド高さ
 * @param stride_w ストライド幅
 * @param pad_top 上パディング
 * @param pad_left 左パディング
 * @param pad_bottom 下パディング
 * @param pad_right 右パディング
 * @return 出力テンソル (flattened)
 */
inline Eigen::MatrixXd averagepool(
    const Eigen::MatrixXd& X,
    int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h = -1, int stride_w = -1,
    int pad_top = 0, int pad_left = 0,
    int pad_bottom = 0, int pad_right = 0) {

    // Default strides to kernel size
    if (stride_h == -1) stride_h = kernel_h;
    if (stride_w == -1) stride_w = kernel_w;

    // Calculate output dimensions
    int out_h = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    int out_w = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

    Eigen::MatrixXd result(C, out_h * out_w);

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < out_h; ++h) {
            for (int w = 0; w < out_w; ++w) {
                int h_start = h * stride_h;
                int w_start = w * stride_w;

                double sum = 0.0;
                int count = 0;

                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int h_idx = h_start + kh - pad_top;
                        int w_idx = w_start + kw - pad_left;

                        // Check bounds (padding areas are counted as 0)
                        if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                            sum += X(c, h_idx * W + w_idx);
                        }
                        count++;
                    }
                }

                result(c, h * out_w + w) = sum / count;
            }
        }
    }

    return result;
}

} // namespace onnx

#endif // ONNX_03_AVERAGEPOOL_HPP
