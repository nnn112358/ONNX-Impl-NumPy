#ifndef ONNX_01_MUL_HPP
#define ONNX_01_MUL_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Mul operator
 *
 * 2つのテンソルの要素ごとの乗算を行う。
 * ブロードキャストをサポート。
 *
 * @param A 入力テンソル1
 * @param B 入力テンソル2
 * @return C: A * B の結果（要素ごとの乗算）
 */
template<typename Derived1, typename Derived2>
auto mul(const Eigen::MatrixBase<Derived1>& A,
         const Eigen::MatrixBase<Derived2>& B) {
    // Same shape - element-wise multiplication
    if (A.rows() == B.rows() && A.cols() == B.cols()) {
        return (A.array() * B.array()).matrix();
    }

    // Broadcasting: B is a row vector (1, cols)
    if (B.rows() == 1 && B.cols() == A.cols()) {
        return (A.array().rowwise() * B.row(0).array()).matrix();
    }

    // Broadcasting: B is a column vector (rows, 1)
    if (B.cols() == 1 && B.rows() == A.rows()) {
        return (A.array().colwise() * B.col(0).array()).matrix();
    }

    // Broadcasting: B is a scalar (1, 1)
    if (B.rows() == 1 && B.cols() == 1) {
        return (A.array() * B(0, 0)).matrix();
    }

    // General broadcasting
    Eigen::MatrixXd result(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            int bi = (B.rows() == 1) ? 0 : i;
            int bj = (B.cols() == 1) ? 0 : j;
            result(i, j) = A(i, j) * B(bi, bj);
        }
    }
    return result;
}

} // namespace onnx

#endif // ONNX_01_MUL_HPP
