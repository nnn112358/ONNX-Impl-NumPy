#include <iostream>
#include <cassert>
#include <cmath>
#include "../01_exp.hpp"

int main() {
    using namespace onnx;

    // Test 1: Basic exponentiation
    Eigen::MatrixXd X(2, 2);
    X << 0, 1,
         2, 3;

    auto Y = exp(X);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << std::exp(0.0), std::exp(1.0),
                 std::exp(2.0), std::exp(3.0);

    assert((Y - expected1).norm() < 1e-10);
    std::cout << "Test 1 (basic exp) passed" << std::endl;

    // Test 2: Negative values
    Eigen::VectorXd X2(4);
    X2 << -1, 0, 1, 2;

    auto Y2 = exp(X2);
    Eigen::VectorXd expected2(4);
    expected2 << std::exp(-1.0), std::exp(0.0), std::exp(1.0), std::exp(2.0);

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (negative values) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
