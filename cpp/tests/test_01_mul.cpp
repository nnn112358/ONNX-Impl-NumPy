#include <iostream>
#include <cassert>
#include "../01_mul.hpp"

int main() {
    using namespace onnx;

    // Test 1: Same shape multiplication
    Eigen::MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    Eigen::MatrixXd B(2, 2);
    B << 5, 6,
         7, 8;

    auto C = mul(A, B);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << 5, 12,
                 21, 32;

    assert((C - expected1).norm() < 1e-10);
    std::cout << "Test 1 (same shape) passed" << std::endl;

    // Test 2: Broadcasting
    Eigen::MatrixXd A2(2, 3);
    A2 << 1, 2, 3,
          4, 5, 6;
    Eigen::MatrixXd B2(1, 3);
    B2 << 2, 3, 4;

    auto C2 = mul(A2, B2);
    Eigen::MatrixXd expected2(2, 3);
    expected2 << 2, 6, 12,
                 8, 15, 24;

    assert((C2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (broadcasting) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
