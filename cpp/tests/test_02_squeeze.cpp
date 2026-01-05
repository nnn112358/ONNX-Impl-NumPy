#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_squeeze.hpp"

int main() {
    using namespace onnx;

    // Test 1: Squeeze (1, 5) -> (5, 1)
    Eigen::MatrixXd data1(1, 5);
    data1 << 1, 2, 3, 4, 5;

    auto squeezed1 = squeeze(data1);
    assert(squeezed1.rows() == 5);
    assert(squeezed1.cols() == 1);

    Eigen::MatrixXd expected1(5, 1);
    expected1 << 1, 2, 3, 4, 5;
    assert((squeezed1 - expected1).norm() < 1e-10);
    std::cout << "Test 1 (1x5 to 5x1) passed" << std::endl;

    // Test 2: Squeeze (1, 3) with axis=0
    Eigen::MatrixXd data2(1, 3);
    data2 << 1, 2, 3;

    auto squeezed2 = squeeze(data2, 0);
    assert(squeezed2.rows() == 3);
    assert(squeezed2.cols() == 1);
    std::cout << "Test 2 (axis=0) passed" << std::endl;

    // Test 3: Squeeze (3, 1) - already column vector
    Eigen::MatrixXd data3(3, 1);
    data3 << 1, 2, 3;

    auto squeezed3 = squeeze(data3);
    assert(squeezed3.rows() == 3);
    assert(squeezed3.cols() == 1);
    assert((squeezed3 - data3).norm() < 1e-10);
    std::cout << "Test 3 (column vector) passed" << std::endl;

    // Test 4: No squeeze needed (2, 3)
    Eigen::MatrixXd data4(2, 3);
    data4 << 1, 2, 3,
             4, 5, 6;

    auto squeezed4 = squeeze(data4);
    assert(squeezed4.rows() == 2);
    assert(squeezed4.cols() == 3);
    assert((squeezed4 - data4).norm() < 1e-10);
    std::cout << "Test 4 (no squeeze) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
