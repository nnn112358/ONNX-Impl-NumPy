#include <iostream>
#include <cassert>
#include <cmath>
#include "../01_sqrt.hpp"

int main() {
    using namespace onnx;

    // Test 1: Perfect squares
    Eigen::MatrixXd X(2, 2);
    X << 1, 4,
         9, 16;

    auto Y = sqrt(X);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << 1, 2,
                 3, 4;

    assert((Y - expected1).norm() < 1e-10);
    std::cout << "Test 1 (perfect squares) passed" << std::endl;

    // Test 2: Various values
    Eigen::VectorXd X2(6);
    X2 << 0, 1, 2, 3, 4, 5;

    auto Y2 = sqrt(X2);
    Eigen::VectorXd expected2(6);
    for (int i = 0; i < 6; ++i) {
        expected2(i) = std::sqrt(X2(i));
    }

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (various values) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
