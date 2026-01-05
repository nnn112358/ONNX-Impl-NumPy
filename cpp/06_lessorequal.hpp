#ifndef ONNX_06_LESSOREQUAL_HPP
#define ONNX_06_LESSOREQUAL_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX LessOrEqual operator
 *
 * 要素ごとの以下比較を行う。
 * ブロードキャストをサポート。
 *
 * @param A 入力テンソル1
 * @param B 入力テンソル2
 * @return C: A <= B の結果（ブール配列）
 */
template<typename Derived1, typename Derived2>
auto lessorequal(const Eigen::MatrixBase<Derived1>& A,
                 const Eigen::MatrixBase<Derived2>& B) {
    // Same shape - direct comparison
    if (A.rows() == B.rows() && A.cols() == B.cols()) {
        return (A.array() <= B.array()).template cast<double>();
    }

    // Broadcasting
    Eigen::MatrixXd result(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            int bi = (B.rows() == 1) ? 0 : i;
            int bj = (B.cols() == 1) ? 0 : j;
            result(i, j) = (A(i, j) <= B(bi, bj)) ? 1.0 : 0.0;
        }
    }
    return result;
}

} // namespace onnx

#endif // ONNX_06_LESSOREQUAL_HPP
