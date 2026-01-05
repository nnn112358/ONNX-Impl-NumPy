#ifndef ONNX_02_SLICE_HPP
#define ONNX_02_SLICE_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Slice operator
 *
 * テンソルの一部を切り出す。
 *
 * @param data 入力テンソル
 * @param starts 各軸の開始インデックスのリスト [row_start, col_start]
 * @param ends 各軸の終了インデックスのリスト [row_end, col_end]
 * @param steps 各軸のステップのリスト [row_step, col_step] (デフォルト: [1, 1])
 * @return output: スライスされたテンソル
 */
template<typename Derived>
Eigen::MatrixXd slice_op(const Eigen::MatrixBase<Derived>& data,
                         const std::vector<int>& starts,
                         const std::vector<int>& ends,
                         const std::vector<int>& steps = {1, 1}) {
    int row_start = starts[0];
    int col_start = starts.size() > 1 ? starts[1] : 0;
    int row_end = ends[0];
    int col_end = ends.size() > 1 ? ends[1] : data.cols();
    int row_step = steps[0];
    int col_step = steps.size() > 1 ? steps[1] : 1;

    // 計算結果のサイズ
    int result_rows = (row_end - row_start + row_step - 1) / row_step;
    int result_cols = (col_end - col_start + col_step - 1) / col_step;

    Eigen::MatrixXd result(result_rows, result_cols);

    int out_row = 0;
    for (int i = row_start; i < row_end; i += row_step) {
        int out_col = 0;
        for (int j = col_start; j < col_end; j += col_step) {
            result(out_row, out_col) = data(i, j);
            out_col++;
        }
        out_row++;
    }

    return result;
}

/**
 * Simple slice with axis parameter
 *
 * @param data 入力テンソル
 * @param start 開始インデックス
 * @param end 終了インデックス
 * @param axis 操作する軸 (0: 行, 1: 列)
 * @param step ステップ (デフォルト: 1)
 * @return output: スライスされたテンソル
 */
template<typename Derived>
Eigen::MatrixXd slice_op(const Eigen::MatrixBase<Derived>& data,
                         int start,
                         int end,
                         int axis = 0,
                         int step = 1) {
    if (axis == 0) {
        // 行方向のスライス
        int result_rows = (end - start + step - 1) / step;
        Eigen::MatrixXd result(result_rows, data.cols());

        int out_row = 0;
        for (int i = start; i < end; i += step) {
            result.row(out_row) = data.row(i);
            out_row++;
        }
        return result;
    } else {
        // 列方向のスライス
        int result_cols = (end - start + step - 1) / step;
        Eigen::MatrixXd result(data.rows(), result_cols);

        int out_col = 0;
        for (int j = start; j < end; j += step) {
            result.col(out_col) = data.col(j);
            out_col++;
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_02_SLICE_HPP
