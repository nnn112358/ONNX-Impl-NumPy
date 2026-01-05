#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "../09_depthtospace.hpp"
#include "../09_spacetodepth.hpp"  // For roundtrip test

int main() {
    using namespace onnx;

    // Test 1: Simple single output channel case
    std::vector<Eigen::MatrixXd> input_channels(4);

    // Channel 0: top-left of each 2x2 block
    input_channels[0] = Eigen::MatrixXd(2, 2);
    input_channels[0] << 0, 2,
                         8, 10;

    // Channel 1: top-right of each 2x2 block
    input_channels[1] = Eigen::MatrixXd(2, 2);
    input_channels[1] << 1, 3,
                         9, 11;

    // Channel 2: bottom-left of each 2x2 block
    input_channels[2] = Eigen::MatrixXd(2, 2);
    input_channels[2] << 4, 6,
                         12, 14;

    // Channel 3: bottom-right of each 2x2 block
    input_channels[3] = Eigen::MatrixXd(2, 2);
    input_channels[3] << 5, 7,
                         13, 15;

    int blocksize = 2;
    auto output = depthtospace_single(input_channels, blocksize);

    // Should produce a 4x4 matrix
    assert(output.rows() == 4);
    assert(output.cols() == 4);

    Eigen::MatrixXd expected(4, 4);
    expected << 0,  1,  2,  3,
                4,  5,  6,  7,
                8,  9, 10, 11,
               12, 13, 14, 15;

    assert((output - expected).norm() < 1e-10);
    std::cout << "Test 1 (single output channel) passed" << std::endl;

    // Test 2: Multi-channel case
    std::vector<Eigen::MatrixXd> input_multi(8);

    // First 4 channels (from first input channel)
    for (int i = 0; i < 4; ++i) {
        input_multi[i] = input_channels[i];
    }

    // Next 4 channels (from second input channel)
    input_multi[4] = Eigen::MatrixXd(2, 2);
    input_multi[4] << 16, 18,
                      24, 26;

    input_multi[5] = Eigen::MatrixXd(2, 2);
    input_multi[5] << 17, 19,
                      25, 27;

    input_multi[6] = Eigen::MatrixXd(2, 2);
    input_multi[6] << 20, 22,
                      28, 30;

    input_multi[7] = Eigen::MatrixXd(2, 2);
    input_multi[7] << 21, 23,
                      29, 31;

    auto output_multi = depthtospace(input_multi, 2);

    // Should produce 2 output channels
    assert(output_multi.size() == 2);
    assert(output_multi[0].rows() == 4);
    assert(output_multi[0].cols() == 4);

    // Verify first output channel
    assert((output_multi[0] - expected).norm() < 1e-10);

    // Verify second output channel
    Eigen::MatrixXd expected2(4, 4);
    expected2 << 16, 17, 18, 19,
                 20, 21, 22, 23,
                 24, 25, 26, 27,
                 28, 29, 30, 31;
    assert((output_multi[1] - expected2).norm() < 1e-10);

    std::cout << "Test 2 (multi-channel) passed" << std::endl;

    // Test 3: Roundtrip test (SpaceToDepth -> DepthToSpace)
    Eigen::MatrixXd original(4, 4);
    original << 0,  1,  2,  3,
                4,  5,  6,  7,
                8,  9, 10, 11,
               12, 13, 14, 15;

    auto s2d_result = spacetodepth(original, 2);
    auto d2s_result = depthtospace_single(s2d_result, 2);

    assert((d2s_result - original).norm() < 1e-10);
    std::cout << "Test 3 (roundtrip) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
