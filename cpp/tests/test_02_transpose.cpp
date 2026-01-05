#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_transpose.hpp"

int main() {
    using namespace onnx;

    // Test 1: 2x3 matrix transpose
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3,
            4, 5, 6;

    auto transposed = transpose(data);

    Eigen::MatrixXd expected(3, 2);
    expected << 1, 4,
                2, 5,
                3, 6;

    assert(transposed.rows() == 3);
    assert(transposed.cols() == 2);
    assert((transposed - expected).norm() < 1e-10);
    std::cout << "Test 1 (2x3 transpose) passed" << std::endl;

    // Test 2: Square matrix transpose
    Eigen::MatrixXd data2(3, 3);
    data2 << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    auto transposed2 = transpose(data2);

    Eigen::MatrixXd expected2(3, 3);
    expected2 << 1, 4, 7,
                 2, 5, 8,
                 3, 6, 9;

    assert((transposed2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (square matrix) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
