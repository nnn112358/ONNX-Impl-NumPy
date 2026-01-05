#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_gru.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple GRU with known dimensions
    int seq_length = 3;
    int input_size = 4;
    int hidden_size = 5;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(seq_length, input_size);
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(3 * hidden_size, input_size);
    Eigen::MatrixXd R = Eigen::MatrixXd::Random(3 * hidden_size, hidden_size);
    Eigen::VectorXd Wb = Eigen::VectorXd::Random(3 * hidden_size);
    Eigen::VectorXd Rb = Eigen::VectorXd::Random(3 * hidden_size);

    auto [Y, Y_h] = gru(X, W, R, &Wb, &Rb);

    // Check output dimensions
    assert(Y.rows() == seq_length);
    assert(Y.cols() == hidden_size);
    assert(Y_h.size() == hidden_size);

    // Check that last output matches final hidden state
    assert((Y.row(seq_length - 1) - Y_h.transpose()).norm() < 1e-10);

    std::cout << "Test 1 (dimensions) passed" << std::endl;

    // Test 2: GRU without bias
    auto [Y2, Y_h2] = gru(X, W, R);

    assert(Y2.rows() == seq_length);
    assert(Y2.cols() == hidden_size);

    std::cout << "Test 2 (no bias) passed" << std::endl;

    // Test 3: With initial state
    Eigen::VectorXd h0 = Eigen::VectorXd::Random(hidden_size);

    auto [Y3, Y_h3] = gru(X, W, R, &Wb, &Rb, &h0);

    assert(Y3.rows() == seq_length);
    assert(Y3.cols() == hidden_size);

    std::cout << "Test 3 (with initial state) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
