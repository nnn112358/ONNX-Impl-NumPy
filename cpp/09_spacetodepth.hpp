#ifndef ONNX_09_SPACETODEPTH_HPP
#define ONNX_09_SPACETODEPTH_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX SpaceToDepth operator
 *
 * 空間次元（H, W）をチャネル次元に再配置する。
 * Simplified for 2D case: treats input as a single channel (H, W) matrix
 * and rearranges it into multiple channels.
 *
 * For a blocksize of 2:
 * (H, W) -> (blocksize^2, H/blocksize, W/blocksize)
 * Returns a vector of matrices representing the output channels.
 *
 * @param input_tensor 入力テンソル (H, W)
 * @param blocksize ブロックサイズ
 * @return 再配置されたテンソル (vector of matrices, each representing a channel)
 */
template<typename Derived>
std::vector<Eigen::MatrixXd> spacetodepth(const Eigen::MatrixBase<Derived>& input_tensor,
                                           int blocksize) {
    int H = input_tensor.rows();
    int W = input_tensor.cols();

    int new_H = H / blocksize;
    int new_W = W / blocksize;
    int new_C = blocksize * blocksize;

    std::vector<Eigen::MatrixXd> output(new_C, Eigen::MatrixXd(new_H, new_W));

    // Rearrange spatial blocks into channels
    for (int c = 0; c < new_C; ++c) {
        int block_h = c / blocksize;
        int block_w = c % blocksize;

        for (int h = 0; h < new_H; ++h) {
            for (int w = 0; w < new_W; ++w) {
                int src_h = h * blocksize + block_h;
                int src_w = w * blocksize + block_w;
                output[c](h, w) = input_tensor(src_h, src_w);
            }
        }
    }

    return output;
}

/**
 * Multi-channel version of SpaceToDepth
 *
 * @param input_channels 入力チャネル (vector of matrices)
 * @param blocksize ブロックサイズ
 * @return 再配置されたテンソル (vector of matrices)
 */
std::vector<Eigen::MatrixXd> spacetodepth_multi(const std::vector<Eigen::MatrixXd>& input_channels,
                                                 int blocksize) {
    if (input_channels.empty()) {
        return std::vector<Eigen::MatrixXd>();
    }

    int C = input_channels.size();
    int H = input_channels[0].rows();
    int W = input_channels[0].cols();

    int new_H = H / blocksize;
    int new_W = W / blocksize;
    int new_C = C * blocksize * blocksize;

    std::vector<Eigen::MatrixXd> output(new_C, Eigen::MatrixXd(new_H, new_W));

    // For each input channel
    for (int c_in = 0; c_in < C; ++c_in) {
        // Rearrange each spatial block
        for (int block_idx = 0; block_idx < blocksize * blocksize; ++block_idx) {
            int block_h = block_idx / blocksize;
            int block_w = block_idx % blocksize;
            int c_out = c_in * blocksize * blocksize + block_idx;

            for (int h = 0; h < new_H; ++h) {
                for (int w = 0; w < new_W; ++w) {
                    int src_h = h * blocksize + block_h;
                    int src_w = w * blocksize + block_w;
                    output[c_out](h, w) = input_channels[c_in](src_h, src_w);
                }
            }
        }
    }

    return output;
}

} // namespace onnx

#endif // ONNX_09_SPACETODEPTH_HPP
