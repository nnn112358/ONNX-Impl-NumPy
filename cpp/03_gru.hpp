#ifndef ONNX_03_GRU_HPP
#define ONNX_03_GRU_HPP

#include <Eigen/Dense>
#include <cmath>
#include <tuple>

namespace onnx {

/**
 * Helper function: sigmoid activation
 */
inline double gru_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * Helper function: tanh activation
 */
inline double gru_tanh(double x) {
    return std::tanh(x);
}

/**
 * ONNX GRU operator
 *
 * GRUセルの順伝播を行う。
 * Simplified implementation for single direction, batch_size=1.
 *
 * @param X 入力テンソル (seq_length x input_size)
 * @param W 入力重み (3*hidden_size x input_size)
 * @param R リカレント重み (3*hidden_size x hidden_size)
 * @param Wb 入力バイアス (3*hidden_size) - optional
 * @param Rb リカレントバイアス (3*hidden_size) - optional
 * @param initial_h 初期隠れ状態 (hidden_size) - optional
 * @return tuple of (Y, Y_h) where:
 *         Y: 出力テンソル (seq_length x hidden_size)
 *         Y_h: 最終隠れ状態 (hidden_size)
 */
inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd> gru(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& R,
    const Eigen::VectorXd* Wb = nullptr,
    const Eigen::VectorXd* Rb = nullptr,
    const Eigen::VectorXd* initial_h = nullptr) {

    int seq_length = X.rows();
    int input_size = X.cols();
    int hidden_size = R.cols();

    // Initialize hidden state
    Eigen::VectorXd h = (initial_h != nullptr) ? *initial_h : Eigen::VectorXd::Zero(hidden_size);

    // Initialize biases
    Eigen::VectorXd wb = (Wb != nullptr) ? *Wb : Eigen::VectorXd::Zero(3 * hidden_size);
    Eigen::VectorXd rb = (Rb != nullptr) ? *Rb : Eigen::VectorXd::Zero(3 * hidden_size);

    Eigen::MatrixXd Y(seq_length, hidden_size);

    // Process each timestep
    for (int t = 0; t < seq_length; ++t) {
        Eigen::VectorXd x_t = X.row(t);

        // Compute input and hidden contributions
        Eigen::VectorXd gates_input = W * x_t + wb;
        Eigen::VectorXd gates_hidden = R * h + rb;

        // Extract gates: [update, reset, candidate]
        Eigen::VectorXd z_gate(hidden_size);  // update gate
        Eigen::VectorXd r_gate(hidden_size);  // reset gate
        Eigen::VectorXd h_tilde(hidden_size); // candidate hidden state

        for (int j = 0; j < hidden_size; ++j) {
            z_gate(j) = gru_sigmoid(gates_input(j) + gates_hidden(j));
            r_gate(j) = gru_sigmoid(gates_input(hidden_size + j) + gates_hidden(hidden_size + j));
        }

        // Compute candidate hidden state with reset gate
        for (int j = 0; j < hidden_size; ++j) {
            h_tilde(j) = gru_tanh(gates_input(2 * hidden_size + j) +
                                  r_gate(j) * gates_hidden(2 * hidden_size + j));
        }

        // Update hidden state
        h = (1.0 - z_gate.array()) * h_tilde.array() + z_gate.array() * h.array();

        Y.row(t) = h;
    }

    return std::make_tuple(Y, h);
}

} // namespace onnx

#endif // ONNX_03_GRU_HPP
