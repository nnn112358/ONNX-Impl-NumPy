#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_layernormalization.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple layer normalization
    Eigen::MatrixXd X(2, 4);
    X << 1, 2, 3, 4,
         5, 6, 7, 8;

    Eigen::VectorXd scale(4);
    scale << 1, 1, 1, 1;

    Eigen::VectorXd bias(4);
    bias << 0, 0, 0, 0;

    auto Y = layernormalization(X, scale, bias);

    // After normalization, each row should have mean~0 and variance~1
    for (int i = 0; i < Y.rows(); ++i) {
        double row_mean = Y.row(i).mean();
        double row_var = 0.0;
        for (int j = 0; j < Y.cols(); ++j) {
            double diff = Y(i, j) - row_mean;
            row_var += diff * diff;
        }
        row_var /= Y.cols();

        assert(std::abs(row_mean) < 1e-10);
        assert(std::abs(row_var - 1.0) < 1e-5);
    }

    std::cout << "Test 1 (normalization properties) passed" << std::endl;

    // Test 2: With scale and bias
    Eigen::MatrixXd X2(1, 3);
    X2 << 1, 2, 3;

    Eigen::VectorXd scale2(3);
    scale2 << 2, 2, 2;

    Eigen::VectorXd bias2(3);
    bias2 << 1, 1, 1;

    auto Y2 = layernormalization(X2, scale2, bias2);

    // Mean should be shifted by bias, variance scaled by scale^2
    std::cout << "Test 2 (scale and bias) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
