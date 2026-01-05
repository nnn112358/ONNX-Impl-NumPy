#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_gather.hpp"

int main() {
    using namespace onnx;

    // Test 1: Gather along axis=0 (rows)
    Eigen::MatrixXd data(3, 3);
    data << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    std::vector<int> indices0 = {0, 2};
    auto result0 = gather(data, indices0, 0);

    Eigen::MatrixXd expected0(2, 3);
    expected0 << 1, 2, 3,
                 7, 8, 9;

    assert(result0.rows() == 2);
    assert(result0.cols() == 3);
    assert((result0 - expected0).norm() < 1e-10);
    std::cout << "Test 1 (gather axis=0) passed" << std::endl;

    // Test 2: Gather along axis=1 (columns)
    std::vector<int> indices1 = {0, 2};
    auto result1 = gather(data, indices1, 1);

    Eigen::MatrixXd expected1(3, 2);
    expected1 << 1, 3,
                 4, 6,
                 7, 9;

    assert(result1.rows() == 3);
    assert(result1.cols() == 2);
    assert((result1 - expected1).norm() < 1e-10);
    std::cout << "Test 2 (gather axis=1) passed" << std::endl;

    // Test 3: Gather with Eigen vector
    Eigen::VectorXi indices2(3);
    indices2 << 1, 0, 2;

    auto result2 = gather(data, indices2, 0);

    Eigen::MatrixXd expected2(3, 3);
    expected2 << 4, 5, 6,
                 1, 2, 3,
                 7, 8, 9;

    assert((result2 - expected2).norm() < 1e-10);
    std::cout << "Test 3 (gather with Eigen vector) passed" << std::endl;

    // Test 4: Gather single index
    std::vector<int> indices3 = {1};
    auto result3 = gather(data, indices3, 0);

    Eigen::MatrixXd expected3(1, 3);
    expected3 << 4, 5, 6;

    assert((result3 - expected3).norm() < 1e-10);
    std::cout << "Test 4 (single index) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
