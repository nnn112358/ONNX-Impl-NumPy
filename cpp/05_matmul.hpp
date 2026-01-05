#ifndef ONNX_05_MATMUL_HPP
#define ONNX_05_MATMUL_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX MatMul operator
 *
 * 行列乗算を行う。
 *
 * @param A 入力行列
 * @param B 入力行列
 * @return C: A * B の結果（行列積）
 */
template<typename Derived1, typename Derived2>
auto matmul(const Eigen::MatrixBase<Derived1>& A,
            const Eigen::MatrixBase<Derived2>& B) {
    return (A * B).eval();
}

} // namespace onnx

#endif // ONNX_05_MATMUL_HPP
