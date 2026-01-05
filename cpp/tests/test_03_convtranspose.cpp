#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_convtranspose.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple 2x2 transpose conv
    // Input: 1 channel, 2x2
    Eigen::MatrixXd X(1, 4);
    X << 1, 2,
         3, 4;

    // Kernel: 1 input channel, 1 output channel, 2x2 kernel
    Eigen::MatrixXd W(1, 4);
    W << 1, 1,
         1, 1;

    // With stride 2, should upsample
    auto Y = convtranspose(X, W, nullptr, 1, 2, 2, 1, 2, 2, 2, 2);

    // Output should be larger (upsampling effect)
    assert(Y.rows() == 1);
    assert(Y.cols() > 4);

    std::cout << "Test 1 (simple transpose conv) passed" << std::endl;

    // Test 2: With bias
    Eigen::VectorXd bias(1);
    bias << 5;

    auto Y2 = convtranspose(X, W, &bias, 1, 2, 2, 1, 2, 2, 2, 2);

    // All outputs should be increased by bias
    assert((Y2.array() - Y.array() - 5.0).abs().maxCoeff() < 1e-10);

    std::cout << "Test 2 (with bias) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
