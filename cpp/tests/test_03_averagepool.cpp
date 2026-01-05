#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_averagepool.hpp"

int main() {
    using namespace onnx;

    // Test: AveragePool with 2x2 kernel, stride 2
    // Input: 1 channel, 4x4
    Eigen::MatrixXd X(1, 16);
    X << 1, 2, 3, 4,
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16;

    auto Y = averagepool(X, 1, 4, 4, 2, 2, 2, 2);

    // Expected output: 2x2 with average values from each 2x2 region
    // Region 1: (1,2,5,6) -> avg = 3.5
    // Region 2: (3,4,7,8) -> avg = 5.5
    // Region 3: (9,10,13,14) -> avg = 11.5
    // Region 4: (11,12,15,16) -> avg = 13.5
    Eigen::MatrixXd expected(1, 4);
    expected << 3.5, 5.5, 11.5, 13.5;

    assert((Y - expected).norm() < 1e-10);
    std::cout << "Test 1 (2x2 kernel, stride 2) passed" << std::endl;

    // Test 2: Simple 2x2 input with 2x2 kernel
    Eigen::MatrixXd X2(1, 4);
    X2 << 1, 2,
          3, 4;

    auto Y2 = averagepool(X2, 1, 2, 2, 2, 2, 2, 2);

    Eigen::MatrixXd expected2(1, 1);
    expected2 << 2.5;

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (simple average) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
