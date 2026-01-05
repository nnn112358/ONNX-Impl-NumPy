#include <iostream>
#include <cassert>
#include <cmath>
#include "../02_resize.hpp"

int main() {
    using namespace onnx;

    // Test 1: Resize with scale using nearest neighbor
    Eigen::MatrixXd X(2, 2);
    X << 1, 2,
         3, 4;

    auto resized1 = resize(X, 2.0, 2.0, "nearest");

    assert(resized1.rows() == 4);
    assert(resized1.cols() == 4);

    Eigen::MatrixXd expected1(4, 4);
    expected1 << 1, 1, 2, 2,
                 1, 1, 2, 2,
                 3, 3, 4, 4,
                 3, 3, 4, 4;

    assert((resized1 - expected1).norm() < 1e-10);
    std::cout << "Test 1 (nearest neighbor 2x scale) passed" << std::endl;

    // Test 2: Resize with target size
    auto resized2 = resize(X, 4, 4, "nearest");

    assert(resized2.rows() == 4);
    assert(resized2.cols() == 4);
    assert((resized2 - expected1).norm() < 1e-10);
    std::cout << "Test 2 (target size) passed" << std::endl;

    // Test 3: Downscale
    Eigen::MatrixXd X3(4, 4);
    X3 << 1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16;

    auto resized3 = resize(X3, 0.5, 0.5, "nearest");

    assert(resized3.rows() == 2);
    assert(resized3.cols() == 2);

    Eigen::MatrixXd expected3(2, 2);
    expected3 << 1, 3,
                 9, 11;

    assert((resized3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (downscale) passed" << std::endl;

    // Test 4: Bilinear interpolation
    Eigen::MatrixXd X4(2, 2);
    X4 << 0, 2,
          2, 4;

    auto resized4 = resize(X4, 3, 3, "linear");

    assert(resized4.rows() == 3);
    assert(resized4.cols() == 3);

    // Center value should be interpolated (approximately 2.0)
    double center_val = resized4(1, 1);
    assert(std::abs(center_val - 2.0) < 0.5);  // Approximate check
    std::cout << "Test 4 (bilinear interpolation) passed" << std::endl;

    // Test 5: Identity resize (scale = 1.0)
    auto resized5 = resize(X, 1.0, 1.0, "nearest");

    assert((resized5 - X).norm() < 1e-10);
    std::cout << "Test 5 (identity resize) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
