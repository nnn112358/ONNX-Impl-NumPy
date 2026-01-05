#ifndef ONNX_05_GEMM_HPP
#define ONNX_05_GEMM_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Gemm operator
 *
 * 一般行列乗算（General Matrix Multiplication）。
 * Y = alpha * A' * B' + beta * C
 *
 * @param A 入力行列
 * @param B 入力行列
 * @param C バイアス行列 (オプション)
 * @param alpha A*Bのスカラー倍数 (デフォルト: 1.0)
 * @param beta Cのスカラー倍数 (デフォルト: 1.0)
 * @param transA Aを転置するか (デフォルト: false)
 * @param transB Bを転置するか (デフォルト: false)
 * @return Y: 結果行列
 */
template<typename Derived1, typename Derived2, typename Derived3>
auto gemm(const Eigen::MatrixBase<Derived1>& A,
          const Eigen::MatrixBase<Derived2>& B,
          const Eigen::MatrixBase<Derived3>& C,
          double alpha = 1.0,
          double beta = 1.0,
          bool transA = false,
          bool transB = false) {
    typedef typename Derived1::Scalar Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y;

    if (transA && transB) {
        Y = alpha * A.transpose() * B.transpose() + beta * C;
    } else if (transA) {
        Y = alpha * A.transpose() * B + beta * C;
    } else if (transB) {
        Y = alpha * A * B.transpose() + beta * C;
    } else {
        Y = alpha * A * B + beta * C;
    }

    return Y;
}

// Overload without C (no bias term)
template<typename Derived1, typename Derived2>
auto gemm(const Eigen::MatrixBase<Derived1>& A,
          const Eigen::MatrixBase<Derived2>& B,
          double alpha = 1.0,
          bool transA = false,
          bool transB = false) {
    typedef typename Derived1::Scalar Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Y;

    if (transA && transB) {
        Y = alpha * A.transpose() * B.transpose();
    } else if (transA) {
        Y = alpha * A.transpose() * B;
    } else if (transB) {
        Y = alpha * A * B.transpose();
    } else {
        Y = alpha * A * B;
    }

    return Y;
}

} // namespace onnx

#endif // ONNX_05_GEMM_HPP
