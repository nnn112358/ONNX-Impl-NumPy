#include <iostream>
#include <cassert>
#include "../01_neg.hpp"

int main() {
    using namespace onnx;

    // Test 1: Integer matrix
    Eigen::MatrixXd X(2, 2);
    X << 1, -2,
         3, -4;

    auto Y = neg(X);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << -1, 2,
                 -3, 4;

    assert((Y - expected1).norm() < 1e-10);
    std::cout << "Test 1 (integer matrix) passed" << std::endl;

    // Test 2: Floating point vector
    Eigen::VectorXd X2(3);
    X2 << 1.5, -2.5, 3.0;

    auto Y2 = neg(X2);
    Eigen::VectorXd expected2(3);
    expected2 << -1.5, 2.5, -3.0;

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (floating point) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
