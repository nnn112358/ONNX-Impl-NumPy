#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_concat.hpp"

int main() {
    using namespace onnx;

    // Test 1: Concat along axis=0 (rows)
    Eigen::MatrixXd A(2, 2);
    A << 1, 2,
         3, 4;

    Eigen::MatrixXd B(2, 2);
    B << 5, 6,
         7, 8;

    Eigen::MatrixXd C(2, 2);
    C << 9, 10,
         11, 12;

    auto result0 = concat(A, B, C, 0);

    Eigen::MatrixXd expected0(6, 2);
    expected0 << 1, 2,
                 3, 4,
                 5, 6,
                 7, 8,
                 9, 10,
                 11, 12;

    assert(result0.rows() == 6);
    assert(result0.cols() == 2);
    assert((result0 - expected0).norm() < 1e-10);
    std::cout << "Test 1 (concat axis=0) passed" << std::endl;

    // Test 2: Concat along axis=1 (columns)
    auto result1 = concat(A, B, C, 1);

    Eigen::MatrixXd expected1(2, 6);
    expected1 << 1, 2, 5, 6, 9, 10,
                 3, 4, 7, 8, 11, 12;

    assert(result1.rows() == 2);
    assert(result1.cols() == 6);
    assert((result1 - expected1).norm() < 1e-10);
    std::cout << "Test 2 (concat axis=1) passed" << std::endl;

    // Test 3: Concat 2 matrices
    auto result2 = concat(A, B, 0);

    Eigen::MatrixXd expected2(4, 2);
    expected2 << 1, 2,
                 3, 4,
                 5, 6,
                 7, 8;

    assert((result2 - expected2).norm() < 1e-10);
    std::cout << "Test 3 (concat 2 matrices) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
