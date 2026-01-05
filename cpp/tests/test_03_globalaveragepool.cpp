#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_globalaveragepool.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple 2 channels, 2x2 spatial
    Eigen::MatrixXd X(2, 4);
    X << 1, 2, 3, 4,    // Channel 0
         5, 6, 7, 8;    // Channel 1

    auto Y = globalaveragepool(X, 2, 2, 2);

    // Expected: Channel 0 avg = (1+2+3+4)/4 = 2.5
    //           Channel 1 avg = (5+6+7+8)/4 = 6.5
    Eigen::MatrixXd expected(2, 1);
    expected << 2.5,
                6.5;

    assert((Y - expected).norm() < 1e-10);
    std::cout << "Test 1 (2 channels, 2x2) passed" << std::endl;

    // Test 2: Single channel, 3x3 spatial
    Eigen::MatrixXd X2(1, 9);
    X2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    auto Y2 = globalaveragepool(X2, 1, 3, 3);

    // Expected: avg = (1+2+3+4+5+6+7+8+9)/9 = 5
    Eigen::MatrixXd expected2(1, 1);
    expected2 << 5.0;

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (1 channel, 3x3) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
