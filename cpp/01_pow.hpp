#ifndef ONNX_01_POW_HPP
#define ONNX_01_POW_HPP

#include <Eigen/Dense>

namespace onnx {

/**
 * ONNX Pow operator
 *
 * 2つのテンソルの要素ごとのべき乗を計算する。
 * X^Y を計算。ブロードキャストをサポート。
 *
 * @param X 底テンソル
 * @param Y 指数テンソル
 * @return Z: X^Y の結果
 */
template<typename Derived1, typename Derived2>
auto pow(const Eigen::MatrixBase<Derived1>& X,
         const Eigen::MatrixBase<Derived2>& Y) {
    // Same shape - element-wise power
    if (X.rows() == Y.rows() && X.cols() == Y.cols()) {
        return X.array().pow(Y.array()).matrix();
    }

    // Broadcasting: Y is a row vector (1, cols)
    if (Y.rows() == 1 && Y.cols() == X.cols()) {
        return (X.array().rowwise() * Y.row(0).array()).unaryExpr(
            [](double x) { return x; }).matrix();
    }

    // Broadcasting: Y is a column vector (rows, 1)
    if (Y.cols() == 1 && Y.rows() == X.rows()) {
        Eigen::MatrixXd result(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                result(i, j) = std::pow(X(i, j), Y(i, 0));
            }
        }
        return result;
    }

    // Broadcasting: Y is a scalar (1, 1)
    if (Y.rows() == 1 && Y.cols() == 1) {
        return X.array().pow(Y(0, 0)).matrix();
    }

    // General broadcasting
    Eigen::MatrixXd result(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); ++i) {
        for (int j = 0; j < X.cols(); ++j) {
            int yi = (Y.rows() == 1) ? 0 : i;
            int yj = (Y.cols() == 1) ? 0 : j;
            result(i, j) = std::pow(X(i, j), Y(yi, yj));
        }
    }
    return result;
}

} // namespace onnx

#endif // ONNX_01_POW_HPP
