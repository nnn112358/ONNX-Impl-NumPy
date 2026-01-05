#include <iostream>
#include <cassert>
#include "../01_clip.hpp"

int main() {
    using namespace onnx;

    // Test 1: Clip with both min and max
    Eigen::MatrixXd X(2, 3);
    X << -5, 0, 5,
         10, 15, 20;

    auto Y = clip(X, 0, 10);
    Eigen::MatrixXd expected1(2, 3);
    expected1 << 0, 0, 5,
                 10, 10, 10;

    assert((Y - expected1).norm() < 1e-10);
    std::cout << "Test 1 (min and max) passed" << std::endl;

    // Test 2: Clip with only min
    Eigen::VectorXd X2(7);
    X2 << -3, -2, -1, 0, 1, 2, 3;

    auto Y2 = clip(X2, 0);
    Eigen::VectorXd expected2(7);
    expected2 << 0, 0, 0, 0, 1, 2, 3;

    assert((Y2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (only min) passed" << std::endl;

    // Test 3: Clip with only max
    auto Y3 = clip(X2, -std::numeric_limits<double>::infinity(), 2);
    Eigen::VectorXd expected3(7);
    expected3 << -3, -2, -1, 0, 1, 2, 2;

    assert((Y3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (only max) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
