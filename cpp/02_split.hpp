#ifndef ONNX_02_SPLIT_HPP
#define ONNX_02_SPLIT_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Split operator
 *
 * テンソルを指定された軸に沿って分割する。
 *
 * @param input_tensor 入力テンソル
 * @param axis 分割する軸 (0: 行方向, 1: 列方向)
 * @param num_outputs 出力数（均等分割の場合）
 * @return outputs: 分割されたテンソルのリスト
 */
template<typename Derived>
std::vector<Eigen::MatrixXd> split(const Eigen::MatrixBase<Derived>& input_tensor,
                                   int axis = 0,
                                   int num_outputs = 2) {
    std::vector<Eigen::MatrixXd> outputs;

    if (axis == 0) {
        // 行方向に分割
        int rows_per_split = input_tensor.rows() / num_outputs;

        for (int i = 0; i < num_outputs; ++i) {
            int start_row = i * rows_per_split;
            int rows = (i == num_outputs - 1) ? (input_tensor.rows() - start_row) : rows_per_split;

            Eigen::MatrixXd split_mat = input_tensor.block(start_row, 0, rows, input_tensor.cols());
            outputs.push_back(split_mat);
        }
    } else {
        // 列方向に分割
        int cols_per_split = input_tensor.cols() / num_outputs;

        for (int i = 0; i < num_outputs; ++i) {
            int start_col = i * cols_per_split;
            int cols = (i == num_outputs - 1) ? (input_tensor.cols() - start_col) : cols_per_split;

            Eigen::MatrixXd split_mat = input_tensor.block(0, start_col, input_tensor.rows(), cols);
            outputs.push_back(split_mat);
        }
    }

    return outputs;
}

/**
 * Split with custom sizes
 *
 * @param input_tensor 入力テンソル
 * @param split_sizes 各分割のサイズのリスト
 * @param axis 分割する軸 (0: 行方向, 1: 列方向)
 * @return outputs: 分割されたテンソルのリスト
 */
template<typename Derived>
std::vector<Eigen::MatrixXd> split(const Eigen::MatrixBase<Derived>& input_tensor,
                                   const std::vector<int>& split_sizes,
                                   int axis = 0) {
    std::vector<Eigen::MatrixXd> outputs;

    if (axis == 0) {
        // 行方向に分割
        int current_row = 0;
        for (int size : split_sizes) {
            Eigen::MatrixXd split_mat = input_tensor.block(current_row, 0, size, input_tensor.cols());
            outputs.push_back(split_mat);
            current_row += size;
        }
    } else {
        // 列方向に分割
        int current_col = 0;
        for (int size : split_sizes) {
            Eigen::MatrixXd split_mat = input_tensor.block(0, current_col, input_tensor.rows(), size);
            outputs.push_back(split_mat);
            current_col += size;
        }
    }

    return outputs;
}

} // namespace onnx

#endif // ONNX_02_SPLIT_HPP
