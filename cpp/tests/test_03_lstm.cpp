#include <iostream>
#include <cassert>
#include <cmath>
#include "../03_lstm.hpp"

int main() {
    using namespace onnx;

    // Test 1: Simple LSTM with known dimensions
    int seq_length = 3;
    int input_size = 4;
    int hidden_size = 5;

    Eigen::MatrixXd X = Eigen::MatrixXd::Random(seq_length, input_size);
    Eigen::MatrixXd W = Eigen::MatrixXd::Random(4 * hidden_size, input_size);
    Eigen::MatrixXd R = Eigen::MatrixXd::Random(4 * hidden_size, hidden_size);
    Eigen::VectorXd Wb = Eigen::VectorXd::Random(4 * hidden_size);
    Eigen::VectorXd Rb = Eigen::VectorXd::Random(4 * hidden_size);

    auto [Y, Y_h, Y_c] = lstm(X, W, R, &Wb, &Rb);

    // Check output dimensions
    assert(Y.rows() == seq_length);
    assert(Y.cols() == hidden_size);
    assert(Y_h.size() == hidden_size);
    assert(Y_c.size() == hidden_size);

    // Check that last output matches final hidden state
    assert((Y.row(seq_length - 1) - Y_h.transpose()).norm() < 1e-10);

    std::cout << "Test 1 (dimensions) passed" << std::endl;

    // Test 2: LSTM without bias
    auto [Y2, Y_h2, Y_c2] = lstm(X, W, R);

    assert(Y2.rows() == seq_length);
    assert(Y2.cols() == hidden_size);

    std::cout << "Test 2 (no bias) passed" << std::endl;

    // Test 3: With initial states
    Eigen::VectorXd h0 = Eigen::VectorXd::Random(hidden_size);
    Eigen::VectorXd c0 = Eigen::VectorXd::Random(hidden_size);

    auto [Y3, Y_h3, Y_c3] = lstm(X, W, R, &Wb, &Rb, &h0, &c0);

    assert(Y3.rows() == seq_length);
    assert(Y3.cols() == hidden_size);

    std::cout << "Test 3 (with initial states) passed" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
