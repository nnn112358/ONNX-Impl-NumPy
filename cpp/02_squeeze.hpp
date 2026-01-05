#ifndef ONNX_02_SQUEEZE_HPP
#define ONNX_02_SQUEEZE_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Squeeze operator
 *
 * サイズが1の次元を削除する。
 * Eigenは2D行列のみをサポートするため、(1, n)を(n, 1)または1Dベクトルに、
 * (n, 1)を(n, 1)または1Dベクトルに変換する。
 *
 * @param data 入力テンソル
 * @param axis 削除する軸 (0 or 1、-1の場合は自動検出)
 * @return squeezed: サイズ1の次元が削除されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd squeeze(const Eigen::MatrixBase<Derived>& data, int axis = -1) {
    int rows = data.rows();
    int cols = data.cols();

    if (axis == -1) {
        // 自動検出：サイズ1の次元を削除
        if (rows == 1 && cols == 1) {
            // (1, 1) -> (1, 1) そのまま
            return data.eval();
        } else if (rows == 1) {
            // (1, n) -> (n, 1)
            Eigen::MatrixXd result(cols, 1);
            for (int i = 0; i < cols; ++i) {
                result(i, 0) = data(0, i);
            }
            return result;
        } else if (cols == 1) {
            // (n, 1) -> そのまま (既に列ベクトル)
            return data.eval();
        } else {
            // どちらもサイズ1でない場合はそのまま
            return data.eval();
        }
    } else if (axis == 0) {
        // 軸0を削除 (行が1の場合)
        if (rows == 1) {
            Eigen::MatrixXd result(cols, 1);
            for (int i = 0; i < cols; ++i) {
                result(i, 0) = data(0, i);
            }
            return result;
        } else {
            return data.eval();
        }
    } else if (axis == 1) {
        // 軸1を削除 (列が1の場合)
        if (cols == 1) {
            // 既に列ベクトルなのでそのまま
            return data.eval();
        } else {
            return data.eval();
        }
    }

    return data.eval();
}

} // namespace onnx

#endif // ONNX_02_SQUEEZE_HPP
