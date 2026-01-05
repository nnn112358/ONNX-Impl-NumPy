#ifndef ONNX_02_RESIZE_HPP
#define ONNX_02_RESIZE_HPP

#include <Eigen/Dense>
#include <cmath>

namespace onnx {

/**
 * ONNX Resize operator
 *
 * テンソルをリサイズする。
 * nearest neighbor または bilinear 補間をサポート。
 *
 * @param X 入力テンソル
 * @param scale_row 行方向のスケール係数
 * @param scale_col 列方向のスケール係数
 * @param mode 補間モード ("nearest" または "linear")
 * @return Y: リサイズされたテンソル
 */
template<typename Derived>
Eigen::MatrixXd resize(const Eigen::MatrixBase<Derived>& X,
                       double scale_row,
                       double scale_col,
                       const std::string& mode = "nearest") {
    int input_rows = X.rows();
    int input_cols = X.cols();

    int output_rows = static_cast<int>(std::round(input_rows * scale_row));
    int output_cols = static_cast<int>(std::round(input_cols * scale_col));

    Eigen::MatrixXd Y(output_rows, output_cols);

    if (mode == "nearest") {
        // Nearest neighbor interpolation
        for (int i = 0; i < output_rows; ++i) {
            for (int j = 0; j < output_cols; ++j) {
                // マッピング元の座標を計算
                int src_row = static_cast<int>(std::floor(i / scale_row));
                int src_col = static_cast<int>(std::floor(j / scale_col));

                // 境界チェック
                src_row = std::min(src_row, input_rows - 1);
                src_col = std::min(src_col, input_cols - 1);

                Y(i, j) = X(src_row, src_col);
            }
        }
    } else if (mode == "linear" || mode == "bilinear") {
        // Bilinear interpolation
        for (int i = 0; i < output_rows; ++i) {
            for (int j = 0; j < output_cols; ++j) {
                // マッピング元の連続座標を計算
                double src_row = i / scale_row;
                double src_col = j / scale_col;

                // 近傍の4点を取得
                int r0 = static_cast<int>(std::floor(src_row));
                int r1 = std::min(r0 + 1, input_rows - 1);
                int c0 = static_cast<int>(std::floor(src_col));
                int c1 = std::min(c0 + 1, input_cols - 1);

                // 補間係数
                double dr = src_row - r0;
                double dc = src_col - c0;

                // バイリニア補間
                double v00 = X(r0, c0);
                double v01 = X(r0, c1);
                double v10 = X(r1, c0);
                double v11 = X(r1, c1);

                double v0 = v00 * (1 - dc) + v01 * dc;
                double v1 = v10 * (1 - dc) + v11 * dc;
                Y(i, j) = v0 * (1 - dr) + v1 * dr;
            }
        }
    } else {
        // デフォルトはnearest
        return resize(X, scale_row, scale_col, "nearest");
    }

    return Y;
}

/**
 * Resize with target sizes
 *
 * @param X 入力テンソル
 * @param target_rows 出力の行数
 * @param target_cols 出力の列数
 * @param mode 補間モード
 * @return Y: リサイズされたテンソル
 */
template<typename Derived>
Eigen::MatrixXd resize(const Eigen::MatrixBase<Derived>& X,
                       int target_rows,
                       int target_cols,
                       const std::string& mode = "nearest") {
    double scale_row = static_cast<double>(target_rows) / X.rows();
    double scale_col = static_cast<double>(target_cols) / X.cols();

    return resize(X, scale_row, scale_col, mode);
}

} // namespace onnx

#endif // ONNX_02_RESIZE_HPP
