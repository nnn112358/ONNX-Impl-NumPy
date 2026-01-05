#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_conv.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple 3x3 conv with 1 input channel, 1 output channel
    // Input: 1 channel, 3x3
    Eigen::MatrixXd X(1, 9);
    X << 1, 2, 3,
         4, 5, 6,
         7, 8, 9;

    // Kernel: 1 output channel, 1 input channel, 2x2 kernel
    // All ones kernel
    Eigen::MatrixXd W(1, 4);
    W << 1, 1, 1, 1;

    // No bias
    auto Y = conv(X, W, nullptr, 1, 3, 3, 1, 2, 2, 1, 1);

    // Expected output: 2x2
    // Top-left: 1+2+4+5 = 12
    // Top-right: 2+3+5+6 = 16
    // Bottom-left: 4+5+7+8 = 24
    // Bottom-right: 5+6+8+9 = 28
    Eigen::MatrixXd expected(1, 4);
    expected << 12, 16, 24, 28;

    assert((Y - expected).norm() < 1e-10);
    std::cout << "Test 1 (simple 2x2 conv) passed" << std::endl;

    // Test 2: With bias
    Eigen::VectorXd bias(1);
    bias << 10;

    auto Y2 = conv(X, W, &bias, 1, 3, 3, 1, 2, 2, 1, 1);

    Eigen::MatrixXd expected2(1, 4);
    expected2 << 22, 26, 34, 38;

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (with bias) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
