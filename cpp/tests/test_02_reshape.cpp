#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_reshape.hpp"

int main() {
    using namespace onnx;

    // Test 1: (2, 4) -> (4, 2)
    Eigen::MatrixXd data(2, 4);
    data << 1, 2, 3, 4,
            5, 6, 7, 8;

    auto reshaped = reshape(data, 4, 2);

    Eigen::MatrixXd expected(4, 2);
    expected << 1, 2,
                3, 4,
                5, 6,
                7, 8;

    assert(reshaped.rows() == 4);
    assert(reshaped.cols() == 2);
    assert((reshaped - expected).norm() < 1e-10);
    std::cout << "Test 1 (2x4 to 4x2) passed" << std::endl;

    // Test 2: (4, 2) -> (2, 4)
    auto reshaped2 = reshape(reshaped, 2, 4);
    assert((reshaped2 - data).norm() < 1e-10);
    std::cout << "Test 2 (4x2 to 2x4) passed" << std::endl;

    // Test 3: (3, 3) -> (1, 9)
    Eigen::MatrixXd data3(3, 3);
    data3 << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;

    auto reshaped3 = reshape(data3, 1, 9);

    Eigen::MatrixXd expected3(1, 9);
    expected3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    assert((reshaped3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (3x3 to 1x9) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
