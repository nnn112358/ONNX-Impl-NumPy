#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_maxpool.hpp"

int main() {
    using namespace onnx;

    // Test: MaxPool with 2x2 kernel, stride 2
    // Input: 1 channel, 4x4
    Eigen::MatrixXd X(1, 16);
    X << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16;

    auto Y = maxpool(X, 1, 4, 4, 2, 2, 2, 2);

    // Expected output: 2x2 with max values from each 2x2 region
    Eigen::MatrixXd expected(1, 4);
    expected << 6, 8, 14, 16;

    assert((Y - expected).norm() < 1e-10);
    std::cout << "Test 1 (2x2 kernel, stride 2) passed" << std::endl;

    // Test 2: With padding
    Eigen::MatrixXd X2(1, 9);
    X2 << 1, 2, 3,
          4, 5, 6,
          7, 8, 9;

    auto Y2 = maxpool(X2, 1, 3, 3, 2, 2, 1, 1, 0, 0, 0, 0);

    std::cout << "Test 2 (with padding) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
