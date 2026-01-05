#ifndef ONNX_03_CONVTRANSPOSE_HPP
#define ONNX_03_CONVTRANSPOSE_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX ConvTranspose operator
 *
 * 転置畳み込み（逆畳み込み）演算を行う。
 * Simplified 2D implementation for (1, C_in, H, W) input.
 *
 * @param X 入力テンソル (C_in x (H*W))
 * @param W 重みテンソル (C_in x (M * kH * kW))
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
inline Eigen::MatrixXd convtranspose(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& W,
    const Eigen::VectorXd* B,
    int C_in, int H, int W_dim, int M, int kH, int kW,
    int stride_h = 1, int stride_w = 1,
    int pad_top = 0, int pad_left = 0,
    int pad_bottom = 0, int pad_right = 0) {

    // Calculate output dimensions
    int out_h = (H - 1) * stride_h - pad_top - pad_bottom + kH;
    int out_w = (W_dim - 1) * stride_w - pad_left - pad_right + kW;

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(M, out_h * out_w);

    // For each input position
    for (int c = 0; c < C_in; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W_dim; ++w) {
                double x_val = X(c, h * W_dim + w);

                // For each output channel
                for (int m = 0; m < M; ++m) {
                    // Apply kernel
                    for (int kh = 0; kh < kH; ++kh) {
                        for (int kw = 0; kw < kW; ++kw) {
                            int h_out = h * stride_h + kh - pad_top;
                            int w_out = w * stride_w + kw - pad_left;

                            // Check bounds
                            if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w) {
                                double w_val = W(c, m * kH * kW + kh * kW + kw);
                                result(m, h_out * out_w + w_out) += x_val * w_val;
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if provided
    if (B != nullptr) {
        for (int m = 0; m < M; ++m) {
            for (int i = 0; i < out_h * out_w; ++i) {
                result(m, i) += (*B)(m);
            }
        }
    }

    return result;
}

} // namespace onnx

#endif // ONNX_03_CONVTRANSPOSE_HPP
