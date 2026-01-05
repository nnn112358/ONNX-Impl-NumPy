#include <iostream>
#include <cassert>
#include <cmath>
#include "../01_pow.hpp"

int main() {
    using namespace onnx;

    // Test 1: Same shape power
    Eigen::MatrixXd X(2, 2);
    X << 2, 3,
         4, 5;
    Eigen::MatrixXd Y(2, 2);
    Y << 2, 2,
         3, 2;

    auto Z = pow(X, Y);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << 4, 9,
                 64, 25;

    assert((Z - expected1).norm() < 1e-10);
    std::cout << "Test 1 (same shape) passed" << std::endl;

    // Test 2: Scalar exponent
    Eigen::MatrixXd X2(1, 3);
    X2 << 2, 3, 4;
    Eigen::MatrixXd Y2(1, 1);
    Y2 << 2;

    auto Z2 = pow(X2, Y2);
    Eigen::MatrixXd expected2(1, 3);
    expected2 << 4, 9, 16;

    assert((Z2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (scalar exponent) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
