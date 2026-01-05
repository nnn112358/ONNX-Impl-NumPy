#ifndef ONNX_02_UNSQUEEZE_HPP
#define ONNX_02_UNSQUEEZE_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Unsqueeze operator
 *
 * 指定した位置にサイズ1の新しい次元を追加する。
 * Eigenは2D行列のみをサポートするため、(n, m)を(1, n*m)または(n*m, 1)に変換する。
 *
 * @param data 入力テンソル
 * @param axis 追加する軸の位置 (0で行方向に1を追加、2で列方向に1を追加)
 * @return unsqueezed: サイズ1の次元が追加されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd unsqueeze(const Eigen::MatrixBase<Derived>& data, int axis) {
    int rows = data.rows();
    int cols = data.cols();

    if (axis == 0) {
        // 軸0に次元追加: (n, m) -> (1, n*m)
        Eigen::MatrixXd result(1, rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(0, i * cols + j) = data(i, j);
            }
        }
        return result;
    } else if (axis == 2 || axis == -1) {
        // 軸2に次元追加: (n, m) -> (n*m, 1)
        Eigen::MatrixXd result(rows * cols, 1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i * cols + j, 0) = data(i, j);
            }
        }
        return result;
    } else {
        // axis == 1 or default: そのまま返す（2D行列として既に2次元）
        return data.eval();
    }
}

} // namespace onnx

#endif // ONNX_02_UNSQUEEZE_HPP
