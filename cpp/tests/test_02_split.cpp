#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_split.hpp"

int main() {
    using namespace onnx;

    // Test 1: Split (4, 3) into 2 parts along axis=0
    Eigen::MatrixXd X(4, 3);
    X << 0, 1, 2,
         3, 4, 5,
         6, 7, 8,
         9, 10, 11;

    auto outputs0 = split(X, 0, 2);

    assert(outputs0.size() == 2);
    assert(outputs0[0].rows() == 2);
    assert(outputs0[0].cols() == 3);
    assert(outputs0[1].rows() == 2);
    assert(outputs0[1].cols() == 3);

    Eigen::MatrixXd expected0_0(2, 3);
    expected0_0 << 0, 1, 2,
                   3, 4, 5;
    assert((outputs0[0] - expected0_0).norm() < 1e-10);

    Eigen::MatrixXd expected0_1(2, 3);
    expected0_1 << 6, 7, 8,
                   9, 10, 11;
    assert((outputs0[1] - expected0_1).norm() < 1e-10);

    std::cout << "Test 1 (split axis=0, num_outputs=2) passed" << std::endl;

    // Test 2: Split (4, 3) with custom sizes [1, 3] along axis=0
    std::vector<int> split_sizes = {1, 3};
    auto outputs1 = split(X, split_sizes, 0);

    assert(outputs1.size() == 2);
    assert(outputs1[0].rows() == 1);
    assert(outputs1[1].rows() == 3);

    Eigen::MatrixXd expected1_0(1, 3);
    expected1_0 << 0, 1, 2;
    assert((outputs1[0] - expected1_0).norm() < 1e-10);

    std::cout << "Test 2 (split with custom sizes) passed" << std::endl;

    // Test 3: Split (3, 6) into 2 parts along axis=1
    Eigen::MatrixXd Y(3, 6);
    Y << 1, 2, 3, 4, 5, 6,
         7, 8, 9, 10, 11, 12,
         13, 14, 15, 16, 17, 18;

    auto outputs2 = split(Y, 1, 2);

    assert(outputs2.size() == 2);
    assert(outputs2[0].rows() == 3);
    assert(outputs2[0].cols() == 3);
    assert(outputs2[1].rows() == 3);
    assert(outputs2[1].cols() == 3);

    Eigen::MatrixXd expected2_0(3, 3);
    expected2_0 << 1, 2, 3,
                   7, 8, 9,
                   13, 14, 15;
    assert((outputs2[0] - expected2_0).norm() < 1e-10);

    std::cout << "Test 3 (split axis=1) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
