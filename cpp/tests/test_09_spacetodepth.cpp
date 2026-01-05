#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../09_spacetodepth.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple single channel case
    Eigen::MatrixXd input(4, 4);
    input << 0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,
            12, 13, 14, 15;

    int blocksize = 2;
    auto output = spacetodepth(input, blocksize);

    // Should produce 4 channels of 2x2
    assert(output.size() == 4);
    assert(output[0].rows() == 2);
    assert(output[0].cols() == 2);

    // Channel 0: top-left of each 2x2 block
    Eigen::MatrixXd expected_ch0(2, 2);
    expected_ch0 << 0, 2,
                    8, 10;
    assert((output[0] - expected_ch0).norm() < 1e-10);

    // Channel 1: top-right of each 2x2 block
    Eigen::MatrixXd expected_ch1(2, 2);
    expected_ch1 << 1, 3,
                    9, 11;
    assert((output[1] - expected_ch1).norm() < 1e-10);

    // Channel 2: bottom-left of each 2x2 block
    Eigen::MatrixXd expected_ch2(2, 2);
    expected_ch2 << 4, 6,
                    12, 14;
    assert((output[2] - expected_ch2).norm() < 1e-10);

    // Channel 3: bottom-right of each 2x2 block
    Eigen::MatrixXd expected_ch3(2, 2);
    expected_ch3 << 5, 7,
                    13, 15;
    assert((output[3] - expected_ch3).norm() < 1e-10);

    std::cout << "Test 1 (single channel) passed" << std::endl;

    // Test 2: Multi-channel case
    std::vector<Eigen::MatrixXd> input_multi(2);
    input_multi[0] = Eigen::MatrixXd(4, 4);
    input_multi[0] << 0,  1,  2,  3,
                      4,  5,  6,  7,
                      8,  9, 10, 11,
                     12, 13, 14, 15;

    input_multi[1] = Eigen::MatrixXd(4, 4);
    input_multi[1] << 16, 17, 18, 19,
                      20, 21, 22, 23,
                      24, 25, 26, 27,
                      28, 29, 30, 31;

    auto output_multi = spacetodepth_multi(input_multi, 2);

    // Should produce 8 channels (2 input * 4 from blocksize^2)
    assert(output_multi.size() == 8);
    assert(output_multi[0].rows() == 2);
    assert(output_multi[0].cols() == 2);

    // Verify first channel from first input
    assert((output_multi[0] - expected_ch0).norm() < 1e-10);

    // Verify first channel from second input (channel index 4)
    Eigen::MatrixXd expected_ch4(2, 2);
    expected_ch4 << 16, 18,
                    24, 26;
    assert((output_multi[4] - expected_ch4).norm() < 1e-10);

    std::cout << "Test 2 (multi-channel) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
