#ifndef ONNX_03_CONV_HPP
#define ONNX_03_CONV_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Conv operator
 *
 * 畳み込み演算を行う。
 * Simplified 2D implementation for (1, C_in, H, W) input, group=1.
 *
 * @param X 入力テンソル (C_in x (H*W))
 * @param W 重みテンソル (M x (C_in * kH * kW))
 * @param B バイアス (M x 1) - optional
 * @param C_in 入力チャネル数
 * @param H 入力高さ
 * @param W_dim 入力幅
 * @param M 出力チャネル数
 * @param kH カーネル高さ
 * @param kW カーネル幅
 * @param stride_h ストライド高さ
 * @param stride_w ストライド幅
 * @param pad_top 上パディング
 * @param pad_left 左パディング
 * @param pad_bottom 下パディング
 * @param pad_right 右パディング
 * @return 出力テンソル (M x (out_h * out_w))
 */
inline Eigen::MatrixXd conv(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd* B,
    int C_in, int H, int W_dim, int M, int kH, int kW,
    int stride_h = 1, int stride_w = 1,
    int pad_top = 0, int pad_left = 0,
    int pad_bottom = 0, int pad_right = 0) {

    // Calculate output dimensions
    int out_h = (H + pad_top + pad_bottom - kH) / stride_h + 1;
    int out_w = (W_dim + pad_left + pad_right - kW) / stride_w + 1;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M, out_h * out_w);

    // For each output channel
    for (int m = 0; m < M; ++m) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int h_start = oh * stride_h;
                int w_start = ow * stride_w;

                double sum = 0.0;

                // Convolve over all input channels
                for (int c = 0; c < C_in; ++c) {
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int h_idx = h_start + kh - pad_top;
                            int w_idx = w_start + kw - pad_left;

                            // Check bounds
                            if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W_dim) {
                                double x_val = X(c, h_idx * W_dim + w_idx);
                                double w_val = W(m, c * kH * kW + kh * kW + kw);
                                sum += x_val * w_val;
                            }
                        }
                    }
                }

                // Add bias if provided
                if (B != nullptr) {
                    sum += (*B)(m);
                }

                result(m, oh * out_w + ow) = sum;
            }
        }
    }

    return result;
}

} // namespace onnx

#endif // ONNX_03_CONV_HPP
