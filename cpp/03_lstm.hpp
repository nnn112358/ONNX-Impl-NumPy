#ifndef ONNX_03_LSTM_HPP
#define ONNX_03_LSTM_HPP

#include <Eigen/Dense>
#include <cmath>
#include <tuple>

namespace onnx {

/**
 * Helper function: sigmoid activation
 */
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * Helper function: tanh activation
 */
inline double tanh_activation(double x) {
    return std::tanh(x);
}

/**
 * ONNX LSTM operator
 *
 * LSTMセルの順伝播を行う。
 * Simplified implementation for single direction, batch_size=1.
 *
 * @param X 入力テンソル (seq_length x input_size)
 * @param W 入力重み (4*hidden_size x input_size)
 * @param R リカレント重み (4*hidden_size x hidden_size)
 * @param Wb 入力バイアス (4*hidden_size) - optional
 * @param Rb リカレントバイアス (4*hidden_size) - optional
 * @param initial_h 初期隠れ状態 (hidden_size) - optional
 * @param initial_c 初期セル状態 (hidden_size) - optional
 * @return tuple of (Y, Y_h, Y_c) where:
 *         Y: 出力テンソル (seq_length x hidden_size)
 *         Y_h: 最終隠れ状態 (hidden_size)
 *         Y_c: 最終セル状態 (hidden_size)
 */
inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> lstm(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& R,
    const Eigen::VectorXd* Wb = nullptr,
    const Eigen::VectorXd* Rb = nullptr,
    const Eigen::VectorXd* initial_h = nullptr,
    const Eigen::VectorXd* initial_c = nullptr) {

    int seq_length = X.rows();
    int input_size = X.cols();
    int hidden_size = R.cols();

    // Initialize hidden and cell states
    Eigen::VectorXd h = (initial_h != nullptr) ? *initial_h : Eigen::VectorXd::Zero(hidden_size);
    Eigen::VectorXd c = (initial_c != nullptr) ? *initial_c : Eigen::VectorXd::Zero(hidden_size);

    // Initialize biases
    Eigen::VectorXd wb = (Wb != nullptr) ? *Wb : Eigen::VectorXd::Zero(4 * hidden_size);
    Eigen::VectorXd rb = (Rb != nullptr) ? *Rb : Eigen::VectorXd::Zero(4 * hidden_size);

    Eigen::MatrixXd Y(seq_length, hidden_size);

    // Process each timestep
    for (int t = 0; t < seq_length; ++t) {
        Eigen::VectorXd x_t = X.row(t);

        // Compute gates: [input, forget, cell, output]
        Eigen::VectorXd gates = W * x_t + R * h + wb + rb;

        // Extract individual gates
        Eigen::VectorXd i_gate(hidden_size);
        Eigen::VectorXd f_gate(hidden_size);
        Eigen::VectorXd g_gate(hidden_size);
        Eigen::VectorXd o_gate(hidden_size);

        for (int j = 0; j < hidden_size; ++j) {
            i_gate(j) = sigmoid(gates(j));                           // input gate
            f_gate(j) = sigmoid(gates(hidden_size + j));            // forget gate
            g_gate(j) = tanh_activation(gates(2 * hidden_size + j)); // cell gate
            o_gate(j) = sigmoid(gates(3 * hidden_size + j));        // output gate
        }

        // Update cell state
        c = f_gate.array() * c.array() + i_gate.array() * g_gate.array();

        // Update hidden state
        for (int j = 0; j < hidden_size; ++j) {
            h(j) = o_gate(j) * tanh_activation(c(j));
        }

        Y.row(t) = h;
    }

    return std::make_tuple(Y, h, c);
}

} // namespace onnx

#endif // ONNX_03_LSTM_HPP
