#ifndef ONNX_01_CLIP_HPP
#define ONNX_01_CLIP_HPP

#include <Eigen/Dense>
#include <limits>

namespace onnx {

/**
 * ONNX Clip operator
 *
 * テンソルの値を指定された範囲にクリップする。
 *
 * @param X 入力テンソル
 * @param min_val 最小値 (省略可能)
 * @param max_val 最大値 (省略可能)
 * @return Y: クリップされた結果
 */
template<typename Derived>
auto clip(const Eigen::MatrixBase<Derived>& X,
          double min_val = -std::numeric_limits<double>::infinity(),
          double max_val = std::numeric_limits<double>::infinity()) {
    return X.array().max(min_val).min(max_val).matrix();
}

} // namespace onnx

#endif // ONNX_01_CLIP_HPP
