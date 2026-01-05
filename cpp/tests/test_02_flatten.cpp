#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_flatten.hpp"

int main() {
    using namespace onnx;

    // Test 1: axis=0 - flatten to (1, 6)
    Eigen::MatrixXd X(2, 3);
    X << 1, 2, 3,
         4, 5, 6;

    auto flat0 = flatten(X, 0);
    assert(flat0.rows() == 1);
    assert(flat0.cols() == 6);

    Eigen::MatrixXd expected0(1, 6);
    expected0 << 1, 2, 3, 4, 5, 6;

    assert((flat0 - expected0).norm() < 1e-10);
    std::cout << "Test 1 (axis=0) passed" << std::endl;

    // Test 2: axis=1 - keep as 2D
    auto flat1 = flatten(X, 1);
    assert(flat1.rows() == 2);
    assert(flat1.cols() == 3);
    assert((flat1 - X).norm() < 1e-10);
    std::cout << "Test 2 (axis=1) passed" << std::endl;

    // Test 3: axis=2 - flatten to (6, 1)
    auto flat2 = flatten(X, 2);
    assert(flat2.rows() == 6);
    assert(flat2.cols() == 1);

    Eigen::MatrixXd expected2(6, 1);
    expected2 << 1, 2, 3, 4, 5, 6;

    assert((flat2 - expected2).norm() < 1e-10);
    std::cout << "Test 3 (axis=2) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
