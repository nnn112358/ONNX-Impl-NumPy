#include <iostream>
#include <cassert>
#include <cmath>
#include "../01_log.hpp"

int main() {
    using namespace onnx;

    // Test 1: Basic logarithm
    Eigen::MatrixXd X(2, 2);
    X << 1, 2,
         M_E, 10;

    auto Y = log(X);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << std::log(1.0), std::log(2.0),
                 std::log(M_E), std::log(10.0);

    assert((Y - expected1).norm() < 1e-10);
    std::cout << "Test 1 (basic log) passed" << std::endl;

    // Test 2: More values
    Eigen::VectorXd X2(5);
    X2 << 1, 2, 3, 4, 5;

    auto Y2 = log(X2);
    Eigen::VectorXd expected2(5);
    for (int i = 0; i < 5; ++i) {
        expected2(i) = std::log(X2(i));
    }

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (vector log) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
