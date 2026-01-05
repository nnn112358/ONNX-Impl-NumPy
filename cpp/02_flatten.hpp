#ifndef ONNX_02_FLATTEN_HPP
#define ONNX_02_FLATTEN_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Flatten operator
 *
 * テンソルを2次元に平坦化する。
 * Eigenは既に2D行列なので、axis=0の場合は(1, rows*cols)に、
 * axis=1の場合は行列をそのまま返す、axis=2の場合は(rows*cols, 1)に変形する。
 *
 * @param input_tensor 入力テンソル
 * @param axis 平坦化の基準となる軸 (0, 1, 2)
 * @return output: 平坦化されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd flatten(const Eigen::MatrixBase<Derived>& input_tensor, int axis = 1) {
    int rows = input_tensor.rows();
    int cols = input_tensor.cols();

    if (axis == 0) {
        // 全要素を1行に平坦化
        Eigen::MatrixXd result(1, rows * cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(0, i * cols + j) = input_tensor(i, j);
            }
        }
        return result;
    } else if (axis == 1) {
        // 2D行列の場合はそのまま返す（既に2次元）
        return input_tensor.eval();
    } else {
        // axis == 2: 全要素を1列に平坦化
        Eigen::MatrixXd result(rows * cols, 1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                result(i * cols + j, 0) = input_tensor(i, j);
            }
        }
        return result;
    }
}

} // namespace onnx

#endif // ONNX_02_FLATTEN_HPP
