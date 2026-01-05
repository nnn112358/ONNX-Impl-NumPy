#ifndef ONNX_10_REVERSESEQUENCE_HPP
#define ONNX_10_REVERSESEQUENCE_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX ReverseSequence operator
 *
 * 各バッチについて、指定された長さまでシーケンスを反転する。
 * Simplified for 2D case (batch_axis and time_axis work on rows/columns)
 *
 * @param input_tensor 入力テンソル
 * @param sequence_lens 各バッチの反転する長さ
 * @param batch_axis バッチ軸 (0: rows, 1: cols)
 * @param time_axis 時間軸（反転する軸） (0: rows, 1: cols)
 * @return 反転されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd reversesequence(const Eigen::MatrixBase<Derived>& input_tensor,
                                 const std::vector<int>& sequence_lens,
                                 int batch_axis = 1,
                                 int time_axis = 0) {
    Eigen::MatrixXd output = input_tensor;

    int batch_size = (batch_axis == 0) ? input_tensor.rows() : input_tensor.cols();

    // Case 1: batch_axis=0 (rows), time_axis=1 (cols)
    // Reverse elements in each row up to sequence_lens[row_idx]
    if (batch_axis == 0 && time_axis == 1) {
        for (int row = 0; row < batch_size; ++row) {
            int seq_len = sequence_lens[row];
            // Reverse the first seq_len elements in this row
            for (int col = 0; col < seq_len / 2; ++col) {
                double temp = output(row, col);
                output(row, col) = output(row, seq_len - 1 - col);
                output(row, seq_len - 1 - col) = temp;
            }
        }
    }
    // Case 2: batch_axis=1 (cols), time_axis=0 (rows)
    // Reverse elements in each column up to sequence_lens[col_idx]
    else if (batch_axis == 1 && time_axis == 0) {
        for (int col = 0; col < batch_size; ++col) {
            int seq_len = sequence_lens[col];
            // Reverse the first seq_len elements in this column
            for (int row = 0; row < seq_len / 2; ++row) {
                double temp = output(row, col);
                output(row, col) = output(seq_len - 1 - row, col);
                output(seq_len - 1 - row, col) = temp;
            }
        }
    }
    // Case 3: batch_axis=0 (rows), time_axis=0 (rows) - same axis
    // Not typically used, but handle it
    else if (batch_axis == 0 && time_axis == 0) {
        // This doesn't make sense in standard usage, treat as column-wise operation
        for (int col = 0; col < input_tensor.cols(); ++col) {
            for (int row = 0; row < batch_size && row < sequence_lens.size(); ++row) {
                int seq_len = sequence_lens[row];
                if (row < seq_len / 2) {
                    double temp = output(row, col);
                    output(row, col) = output(seq_len - 1 - row, col);
                    output(seq_len - 1 - row, col) = temp;
                }
            }
        }
    }
    // Case 4: batch_axis=1 (cols), time_axis=1 (cols) - same axis
    // Not typically used, but handle it
    else if (batch_axis == 1 && time_axis == 1) {
        // This doesn't make sense in standard usage, treat as row-wise operation
        for (int row = 0; row < input_tensor.rows(); ++row) {
            for (int col = 0; col < batch_size && col < sequence_lens.size(); ++col) {
                int seq_len = sequence_lens[col];
                if (col < seq_len / 2) {
                    double temp = output(row, col);
                    output(row, col) = output(row, seq_len - 1 - col);
                    output(row, seq_len - 1 - col) = temp;
                }
            }
        }
    }

    return output;
}

} // namespace onnx

#endif // ONNX_10_REVERSESEQUENCE_HPP
