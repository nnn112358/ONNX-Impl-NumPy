#ifndef ONNX_02_RESHAPE_HPP
#define ONNX_02_RESHAPE_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Reshape operator
 *
 * テンソルを新しい形状に変形する。
 * Eigenは2D行列のみをサポートするため、2D形状への変形を行う。
 *
 * @param data 入力テンソル
 * @param rows 新しい行数
 * @param cols 新しい列数
 * @return reshaped: 変形されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd reshape(const Eigen::MatrixBase<Derived>& data, int rows, int cols) {
    // データを1次元配列として扱い、新しい形状に再構成
    Eigen::MatrixXd reshaped(rows, cols);

    int total_elements = data.rows() * data.cols();

    for (int i = 0; i < total_elements; ++i) {
        int old_row = i / data.cols();
        int old_col = i % data.cols();
        int new_row = i / cols;
        int new_col = i % cols;
        reshaped(new_row, new_col) = data(old_row, old_col);
    }

    return reshaped;
}

} // namespace onnx

#endif // ONNX_02_RESHAPE_HPP
