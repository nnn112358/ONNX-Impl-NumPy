#ifndef ONNX_02_CONCAT_HPP
#define ONNX_02_CONCAT_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Concat operator
 *
 * 複数のテンソルを指定された軸に沿って連結する。
 *
 * @param tensors 連結するテンソルのリスト
 * @param axis 連結する軸 (0: 行方向, 1: 列方向)
 * @return result: 連結されたテンソル
 */
template<typename Derived>
Eigen::MatrixXd concat(const std::vector<Eigen::MatrixBase<Derived>*>& tensors, int axis = 0) {
    if (tensors.empty()) {
        return Eigen::MatrixXd();
    }

    if (axis == 0) {
        // 行方向に連結
        int total_rows = 0;
        int cols = tensors[0]->cols();

        // 総行数を計算
        for (const auto& tensor : tensors) {
            total_rows += tensor->rows();
        }

        Eigen::MatrixXd result(total_rows, cols);
        int current_row = 0;

        for (const auto& tensor : tensors) {
            int rows = tensor->rows();
            result.block(current_row, 0, rows, cols) = *tensor;
            current_row += rows;
        }

        return result;
    } else {
        // 列方向に連結
        int rows = tensors[0]->rows();
        int total_cols = 0;

        // 総列数を計算
        for (const auto& tensor : tensors) {
            total_cols += tensor->cols();
        }

        Eigen::MatrixXd result(rows, total_cols);
        int current_col = 0;

        for (const auto& tensor : tensors) {
            int cols = tensor->cols();
            result.block(0, current_col, rows, cols) = *tensor;
            current_col += cols;
        }

        return result;
    }
}

// Convenience overload for 2 matrices
template<typename Derived1, typename Derived2>
Eigen::MatrixXd concat(const Eigen::MatrixBase<Derived1>& A,
                       const Eigen::MatrixBase<Derived2>& B,
                       int axis = 0) {
    if (axis == 0) {
        // 行方向に連結
        Eigen::MatrixXd result(A.rows() + B.rows(), A.cols());
        result << A, B;
        return result;
    } else {
        // 列方向に連結
        Eigen::MatrixXd result(A.rows(), A.cols() + B.cols());
        result << A, B;
        return result;
    }
}

// Convenience overload for 3 matrices
template<typename Derived1, typename Derived2, typename Derived3>
Eigen::MatrixXd concat(const Eigen::MatrixBase<Derived1>& A,
                       const Eigen::MatrixBase<Derived2>& B,
                       const Eigen::MatrixBase<Derived3>& C,
                       int axis = 0) {
    if (axis == 0) {
        // 行方向に連結
        Eigen::MatrixXd result(A.rows() + B.rows() + C.rows(), A.cols());
        result << A, B, C;
        return result;
    } else {
        // 列方向に連結
        Eigen::MatrixXd result(A.rows(), A.cols() + B.cols() + C.cols());
        result << A, B, C;
        return result;
    }
}

} // namespace onnx

#endif // ONNX_02_CONCAT_HPP
