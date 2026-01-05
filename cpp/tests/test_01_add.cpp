#include <iostream>
#include <cassert>
#include <cmath>
#include "../01_add.hpp"

int main() {
    using namespace onnx;

    // Test 1: Same shape addition
    Eigen::MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;
    Eigen::MatrixXd B(2, 2);
    B << 5, 6,
         7, 8;

    auto C = add(A, B);
    Eigen::MatrixXd expected1(2, 2);
    expected1 << 6, 8,
                 10, 12;

    assert((C - expected1).norm() < 1e-10);
    std::cout << "Test 1 (same shape) passed" << std::endl;

    // Test 2: Broadcasting with row vector
    Eigen::MatrixXd A2(2, 3);
    A2 << 1, 2, 3,
          4, 5, 6;
    Eigen::MatrixXd B2(1, 3);
    B2 << 10, 20, 30;

    auto C2 = add(A2, B2);
    Eigen::MatrixXd expected2(2, 3);
    expected2 << 11, 22, 33,
                 14, 25, 36;

    assert((C2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (broadcasting) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
