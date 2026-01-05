#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_scatternd.hpp"

int main() {
    using namespace onnx;

    // Test 1: ScatterND on 4x4 zeros matrix
    Eigen::MatrixXd data1 = Eigen::MatrixXd::Zero(4, 4);

    Eigen::MatrixXi indices1(3, 2);
    indices1 << 0, 1,
                1, 2,
                2, 3;

    Eigen::VectorXd updates1(3);
    updates1 << 1, 2, 3;

    auto result1 = scatternd(data1, indices1, updates1);

    Eigen::MatrixXd expected1 = Eigen::MatrixXd::Zero(4, 4);
    expected1(0, 1) = 1;
    expected1(1, 2) = 2;
    expected1(2, 3) = 3;

    assert((result1 - expected1).norm() < 1e-10);
    std::cout << "Test 1 (basic scatternd) passed" << std::endl;

    // Test 2: Update diagonal elements
    Eigen::MatrixXd data2 = Eigen::MatrixXd::Ones(3, 3) * 5;

    Eigen::MatrixXi indices2(3, 2);
    indices2 << 0, 0,
                1, 1,
                2, 2;

    Eigen::VectorXd updates2(3);
    updates2 << 10, 20, 30;

    auto result2 = scatternd(data2, indices2, updates2);

    Eigen::MatrixXd expected2(3, 3);
    expected2 << 10, 5, 5,
                 5, 20, 5,
                 5, 5, 30;

    assert((result2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (diagonal update) passed" << std::endl;

    // Test 3: Using vector of pairs
    Eigen::MatrixXd data3 = Eigen::MatrixXd::Zero(3, 3);

    std::vector<std::pair<int, int>> indices3 = {{0, 0}, {1, 1}, {2, 2}};
    std::vector<double> updates3 = {1.0, 2.0, 3.0};

    auto result3 = scatternd(data3, indices3, updates3);

    Eigen::MatrixXd expected3 = Eigen::MatrixXd::Zero(3, 3);
    expected3(0, 0) = 1.0;
    expected3(1, 1) = 2.0;
    expected3(2, 2) = 3.0;

    assert((result3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (vector interface) passed" << std::endl;

    // Test 4: Single element update
    Eigen::MatrixXd data4(2, 3);
    data4 << 1, 2, 3,
             4, 5, 6;

    Eigen::MatrixXi indices4(1, 2);
    indices4 << 1, 2;

    Eigen::VectorXd updates4(1);
    updates4 << 99;

    auto result4 = scatternd(data4, indices4, updates4);

    Eigen::MatrixXd expected4(2, 3);
    expected4 << 1, 2, 3,
                 4, 5, 99;

    assert((result4 - expected4).norm() < 1e-10);
    std::cout << "Test 4 (single update) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
