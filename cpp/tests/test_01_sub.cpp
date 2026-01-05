#include <iostream>
#include <cassert>
#include "../01_sub.hpp"

int main() {
    using namespace onnx;

    // Test 1: Same shape subtraction
    Eigen::MatrixXd A(2, 2);
    A << 10, 20,
         30, 40;
    Eigen::MatrixXd B(2, 2);
    B << 1, 2,
         3, 4;

    auto C = sub(A, B);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << 9, 18,
                 27, 36;

    assert((C - expected1).norm() < 1e-10);
    std::cout << "Test 1 (same shape) passed" << std::endl;

    // Test 2: Broadcasting
    Eigen::MatrixXd A2(1, 3);
    A2 << 100, 200, 300;
    Eigen::MatrixXd B2(1, 1);
    B2 << 50;

    auto C2 = sub(A2, B2);
    Eigen::MatrixXd expected2(1, 3);
    expected2 << 50, 150, 250;

    assert((C2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (broadcasting) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
