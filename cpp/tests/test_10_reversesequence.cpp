#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../10_reversesequence.hpp"

int main() {
    using namespace onnx;

    // Test 1: batch_axis=0 (rows), time_axis=1 (cols)
    // Reverse elements in each row
    Eigen::MatrixXd input1(3, 4);
    input1 << 1, 2, 3, 4,
              5, 6, 7, 8,
              9, 10, 11, 12;

    std::vector<int> sequence_lens1 = {3, 2, 4};

    auto output1 = reversesequence(input1, sequence_lens1, 0, 1);

    Eigen::MatrixXd expected1(3, 4);
    expected1 << 3, 2, 1, 4,     // Row 0: reverse first 3 elements [1,2,3,4] -> [3,2,1,4]
                 6, 5, 7, 8,     // Row 1: reverse first 2 elements [5,6,7,8] -> [6,5,7,8]
                 12, 11, 10, 9;  // Row 2: reverse first 4 elements [9,10,11,12] -> [12,11,10,9]

    assert((output1 - expected1).norm() < 1e-10);
    std::cout << "Test 1 (batch_axis=0, time_axis=1) passed" << std::endl;

    // Test 2: batch_axis=1 (cols), time_axis=0 (rows)
    // Reverse elements in each column
    Eigen::MatrixXd input2(4, 3);
    input2 << 1,  2,  3,
              4,  5,  6,
              7,  8,  9,
              10, 11, 12;

    std::vector<int> sequence_lens2 = {3, 2, 4};

    auto output2 = reversesequence(input2, sequence_lens2, 1, 0);

    Eigen::MatrixXd expected2(4, 3);
    expected2 << 7,  5,  10,  // Col 0: reverse first 3 [1,4,7,10] -> [7,4,1,10]
                 4,  2,  9,   // Col 1: reverse first 2 [2,5,8,11] -> [5,2,8,11]
                 1,  8,  6,   // Col 2: reverse first 4 [3,6,9,12] -> [12,9,6,3]
                 10, 11, 3;

    assert((output2 - expected2).norm() < 1e-10);
    std::cout << "Test 2 (batch_axis=1, time_axis=0) passed" << std::endl;

    // Test 3: Simple 2x2 matrix
    Eigen::MatrixXd input3(2, 2);
    input3 << 1, 2,
              3, 4;

    std::vector<int> sequence_lens3 = {2, 2};

    auto output3 = reversesequence(input3, sequence_lens3, 0, 1);

    Eigen::MatrixXd expected3(2, 2);
    expected3 << 2, 1,
                 4, 3;

    assert((output3 - expected3).norm() < 1e-10);
    std::cout << "Test 3 (simple 2x2) passed" << std::endl;

    // Test 4: Partial reversal
    Eigen::MatrixXd input4(2, 4);
    input4 << 1, 2, 3, 4,
              5, 6, 7, 8;

    std::vector<int> sequence_lens4 = {2, 3};

    auto output4 = reversesequence(input4, sequence_lens4, 0, 1);

    Eigen::MatrixXd expected4(2, 4);
    expected4 << 2, 1, 3, 4,  // Row 0: reverse first 2 [1,2,3,4] -> [2,1,3,4]
                 7, 6, 5, 8;  // Row 1: reverse first 3 [5,6,7,8] -> [7,6,5,8]

    assert((output4 - expected4).norm() < 1e-10);
    std::cout << "Test 4 (partial reversal) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
