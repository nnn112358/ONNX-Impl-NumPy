#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_slice.hpp"

int main() {
    using namespace onnx;

    // Test 1: Basic slice with starts and ends
    Eigen::MatrixXd data(4, 6);
    data << 0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23;

    auto result1 = slice_op(data, {1, 2}, {3, 5});

    Eigen::MatrixXd expected1(2, 3);
    expected1 << 8, 9, 10,
                 14, 15, 16;

    assert(result1.rows() == 2);
    assert(result1.cols() == 3);
    assert((result1 - expected1).norm() < 1e-10);
    std::cout << "Test 1 (basic slice) passed" << std::endl;

    // Test 2: Slice with steps
    auto result2 = slice_op(data, {0, 0}, {4, 6}, {2, 2});

    Eigen::MatrixXd expected2(2, 3);
    expected2 << 0, 2, 4,
                 12, 14, 16;

    assert(result2.rows() == 2);
    assert(result2.cols() == 3);
    assert((result2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (slice with steps) passed" << std::endl;

    // Test 3: Simple slice along axis=0
    auto result3 = slice_op(data, 1, 3, 0);

    Eigen::MatrixXd expected3(2, 6);
    expected3 << 6, 7, 8, 9, 10, 11,
                 12, 13, 14, 15, 16, 17;

    assert((result3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (slice axis=0) passed" << std::endl;

    // Test 4: Simple slice along axis=1
    auto result4 = slice_op(data, 1, 4, 1);

    Eigen::MatrixXd expected4(4, 3);
    expected4 << 1, 2, 3,
                 7, 8, 9,
                 13, 14, 15,
                 19, 20, 21;

    assert((result4 - expected4).norm() < 1e-10);
    std::cout << "Test 4 (slice axis=1) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
