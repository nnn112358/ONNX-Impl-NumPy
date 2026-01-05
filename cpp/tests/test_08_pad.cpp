#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../08_pad.hpp"

int main() {
    using namespace onnx;

    // Test 1: Constant mode padding
    Eigen::MatrixXd data(2, 2);
    data << 1, 2,
            3, 4;

    std::vector<int> pads = {1, 1, 1, 1};  // top, left, bottom, right
    auto result = pad(data, pads, "constant", 0.0);

    Eigen::MatrixXd expected1(4, 4);
    expected1 << 0, 0, 0, 0,
                 0, 1, 2, 0,
                 0, 3, 4, 0,
                 0, 0, 0, 0;

    assert((result - expected1).norm() < 1e-10);
    std::cout << "Test 1 (constant mode) passed" << std::endl;

    // Test 2: Reflect mode padding
    auto result_reflect = pad(data, pads, "reflect");

    Eigen::MatrixXd expected2(4, 4);
    expected2 << 4, 3, 4, 3,
                 2, 1, 2, 1,
                 4, 3, 4, 3,
                 2, 1, 2, 1;

    assert((result_reflect - expected2).norm() < 1e-10);
    std::cout << "Test 2 (reflect mode) passed" << std::endl;

    // Test 3: Edge mode padding
    auto result_edge = pad(data, pads, "edge");

    Eigen::MatrixXd expected3(4, 4);
    expected3 << 1, 1, 2, 2,
                 1, 1, 2, 2,
                 3, 3, 4, 4,
                 3, 3, 4, 4;

    assert((result_edge - expected3).norm() < 1e-10);
    std::cout << "Test 3 (edge mode) passed" << std::endl;

    // Test 4: Asymmetric padding
    Eigen::MatrixXd data2(2, 3);
    data2 << 1, 2, 3,
             4, 5, 6;

    std::vector<int> pads2 = {0, 1, 2, 1};  // top=0, left=1, bottom=2, right=1
    auto result2 = pad(data2, pads2, "constant", 9.0);

    assert(result2.rows() == 4);
    assert(result2.cols() == 5);
    assert(std::abs(result2(0, 0) - 9.0) < 1e-10);
    assert(std::abs(result2(0, 1) - 1.0) < 1e-10);
    assert(std::abs(result2(1, 1) - 4.0) < 1e-10);
    std::cout << "Test 4 (asymmetric padding) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
