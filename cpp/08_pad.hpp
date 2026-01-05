#ifndef ONNX_08_PAD_HPP
#define ONNX_08_PAD_HPP

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace onnx {

/**
 * ONNX Pad operator
 *
 * テンソルにパディングを追加する。
 *
 * @param data 入力テンソル
 * @param pads パディング量 [top, left, bottom, right] for 2D
 * @param mode パディングモード ("constant", "reflect", "edge")
 * @param constant_value constantモード時の値 (デフォルト: 0)
 * @return パディングされたテンソル
 */
template<typename Derived>
Eigen::MatrixXd pad(const Eigen::MatrixBase<Derived>& data,
                    const std::vector<int>& pads,
                    const std::string& mode = "constant",
                    double constant_value = 0.0) {
    typedef typename Derived::Scalar Scalar;

    int rows = data.rows();
    int cols = data.cols();

    // pads: [top, left, bottom, right] for 2D matrix
    int pad_top = pads[0];
    int pad_left = pads[1];
    int pad_bottom = pads[2];
    int pad_right = pads[3];

    int new_rows = rows + pad_top + pad_bottom;
    int new_cols = cols + pad_left + pad_right;

    Eigen::MatrixXd result(new_rows, new_cols);

    if (mode == "constant") {
        // Fill with constant value
        result.setConstant(constant_value);

        // Copy original data
        result.block(pad_top, pad_left, rows, cols) = data;

    } else if (mode == "reflect") {
        // Fill center with original data
        result.block(pad_top, pad_left, rows, cols) = data;

        // Reflect top
        for (int i = 0; i < pad_top; ++i) {
            int src_row = pad_top - i;
            for (int j = pad_left; j < pad_left + cols; ++j) {
                result(i, j) = result(src_row, j);
            }
        }

        // Reflect bottom
        for (int i = 0; i < pad_bottom; ++i) {
            int src_row = pad_top + rows - 2 - i;
            for (int j = pad_left; j < pad_left + cols; ++j) {
                result(pad_top + rows + i, j) = result(src_row, j);
            }
        }

        // Reflect left
        for (int i = 0; i < new_rows; ++i) {
            for (int j = 0; j < pad_left; ++j) {
                int src_col = pad_left + (pad_left - j);
                if (src_col < pad_left + cols) {
                    result(i, j) = result(i, src_col);
                }
            }
        }

        // Reflect right
        for (int i = 0; i < new_rows; ++i) {
            for (int j = 0; j < pad_right; ++j) {
                int src_col = pad_left + cols - 2 - j;
                if (src_col >= pad_left) {
                    result(i, pad_left + cols + j) = result(i, src_col);
                }
            }
        }

    } else if (mode == "edge") {
        // Fill center with original data
        result.block(pad_top, pad_left, rows, cols) = data;

        // Edge top
        for (int i = 0; i < pad_top; ++i) {
            for (int j = pad_left; j < pad_left + cols; ++j) {
                result(i, j) = data(0, j - pad_left);
            }
        }

        // Edge bottom
        for (int i = 0; i < pad_bottom; ++i) {
            for (int j = pad_left; j < pad_left + cols; ++j) {
                result(pad_top + rows + i, j) = data(rows - 1, j - pad_left);
            }
        }

        // Edge left
        for (int i = 0; i < new_rows; ++i) {
            for (int j = 0; j < pad_left; ++j) {
                int src_row = i - pad_top;
                if (src_row < 0) src_row = 0;
                if (src_row >= rows) src_row = rows - 1;
                result(i, j) = data(src_row, 0);
            }
        }

        // Edge right
        for (int i = 0; i < new_rows; ++i) {
            for (int j = 0; j < pad_right; ++j) {
                int src_row = i - pad_top;
                if (src_row < 0) src_row = 0;
                if (src_row >= rows) src_row = rows - 1;
                result(i, pad_left + cols + j) = data(src_row, cols - 1);
            }
        }
    } else {
        // Default to constant mode
        result.setConstant(constant_value);
        result.block(pad_top, pad_left, rows, cols) = data;
    }

    return result;
}

} // namespace onnx

#endif // ONNX_08_PAD_HPP
