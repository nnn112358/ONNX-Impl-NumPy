#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_unsqueeze.hpp"

int main() {
    using namespace onnx;

    // Test 1: Unsqueeze (2, 3) with axis=0 -> (1, 6)
    Eigen::MatrixXd data(2, 3);
    data << 1, 2, 3,
            4, 5, 6;

    auto unsqueezed0 = unsqueeze(data, 0);
    assert(unsqueezed0.rows() == 1);
    assert(unsqueezed0.cols() == 6);

    Eigen::MatrixXd expected0(1, 6);
    expected0 << 1, 2, 3, 4, 5, 6;
    assert((unsqueezed0 - expected0).norm() < 1e-10);
    std::cout << "Test 1 (axis=0) passed" << std::endl;

    // Test 2: Unsqueeze (2, 3) with axis=2 -> (6, 1)
    auto unsqueezed2 = unsqueeze(data, 2);
    assert(unsqueezed2.rows() == 6);
    assert(unsqueezed2.cols() == 1);

    Eigen::MatrixXd expected2(6, 1);
    expected2 << 1, 2, 3, 4, 5, 6;
    assert((unsqueezed2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (axis=2) passed" << std::endl;

    // Test 3: Unsqueeze (3, 1) with axis=0 -> (1, 3)
    Eigen::MatrixXd data3(3, 1);
    data3 << 1, 2, 3;

    auto unsqueezed3 = unsqueeze(data3, 0);
    assert(unsqueezed3.rows() == 1);
    assert(unsqueezed3.cols() == 3);

    Eigen::MatrixXd expected3(1, 3);
    expected3 << 1, 2, 3;
    assert((unsqueezed3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (column to row) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
