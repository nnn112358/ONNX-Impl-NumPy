#ifndef ONNX_09_DEPTHTOSPACE_HPP
#define ONNX_09_DEPTHTOSPACE_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX DepthToSpace operator
 *
 * チャネル次元を空間次元（H, W）に再配置する。
 * Takes multiple channel matrices and rearranges them into a larger spatial matrix.
 *
 * For a blocksize of 2:
 * (blocksize^2 channels, H, W) -> (1 channel, H*blocksize, W*blocksize)
 *
 * @param input_channels 入力チャネル (vector of matrices)
 * @param blocksize ブロックサイズ
 * @return 再配置されたテンソル (single matrix)
 */
std::vector<Eigen::MatrixXd> depthtospace(const std::vector<Eigen::MatrixXd>& input_channels,
                                           int blocksize) {
    if (input_channels.empty()) {
        return std::vector<Eigen::MatrixXd>();
    }

    int C = input_channels.size();
    int H = input_channels[0].rows();
    int W = input_channels[0].cols();

    int new_C = C / (blocksize * blocksize);
    int new_H = H * blocksize;
    int new_W = W * blocksize;

    std::vector<Eigen::MatrixXd> output(new_C, Eigen::MatrixXd(new_H, new_W));

    // For each output channel
    for (int c_out = 0; c_out < new_C; ++c_out) {
        // Rearrange blocks from input channels
        for (int block_idx = 0; block_idx < blocksize * blocksize; ++block_idx) {
            int block_h = block_idx / blocksize;
            int block_w = block_idx % blocksize;
            int c_in = c_out * blocksize * blocksize + block_idx;

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int dst_h = h * blocksize + block_h;
                    int dst_w = w * blocksize + block_w;
                    output[c_out](dst_h, dst_w) = input_channels[c_in](h, w);
                }
            }
        }
    }

    return output;
}

/**
 * Single output channel version of DepthToSpace
 *
 * @param input_channels 入力チャネル (vector of matrices, should have blocksize^2 elements)
 * @param blocksize ブロックサイズ
 * @return 再配置されたテンソル (single matrix)
 */
template<typename T = Eigen::MatrixXd>
Eigen::MatrixXd depthtospace_single(const std::vector<Eigen::MatrixXd>& input_channels,
                                     int blocksize) {
    if (input_channels.empty()) {
        return Eigen::MatrixXd();
    }

    int H = input_channels[0].rows();
    int W = input_channels[0].cols();
    int new_H = H * blocksize;
    int new_W = W * blocksize;

    Eigen::MatrixXd output(new_H, new_W);

    // Rearrange blocks from input channels
    for (int c = 0; c < input_channels.size(); ++c) {
        int block_h = c / blocksize;
        int block_w = c % blocksize;

        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int dst_h = h * blocksize + block_h;
                int dst_w = w * blocksize + block_w;
                output(dst_h, dst_w) = input_channels[c](h, w);
            }
        }
    }

    return output;
}

} // namespace onnx

#endif // ONNX_09_DEPTHTOSPACE_HPP
