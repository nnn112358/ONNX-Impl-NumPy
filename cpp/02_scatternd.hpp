#ifndef ONNX_02_SCATTERND_HPP
#define ONNX_02_SCATTERND_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX ScatterND operator
 *
 * インデックスで指定された位置に更新値を散布する。
 *
 * @param data 入力テンソル (ベーステンソル)
 * @param indices 更新する位置のインデックス (各行が1つの位置を表す)
 * @param updates 更新値
 * @return output: 更新されたテンソル
 */
template<typename Derived1, typename Derived2, typename Derived3>
Eigen::MatrixXd scatternd(const Eigen::MatrixBase<Derived1>& data,
                          const Eigen::MatrixBase<Derived2>& indices,
                          const Eigen::MatrixBase<Derived3>& updates) {
    Eigen::MatrixXd output = data;

    // indices の各行は [row_idx, col_idx] を表す
    for (int i = 0; i < indices.rows(); ++i) {
        int row_idx = static_cast<int>(indices(i, 0));

        if (indices.cols() == 1) {
            // 1D インデックス: データ全体を平坦化して更新
            int total_cols = data.cols();
            int actual_row = row_idx / total_cols;
            int actual_col = row_idx % total_cols;
            output(actual_row, actual_col) = updates(i);
        } else {
            // 2D インデックス
            int col_idx = static_cast<int>(indices(i, 1));
            output(row_idx, col_idx) = updates(i);
        }
    }

    return output;
}

/**
 * ScatterND with vector indices (for convenience)
 *
 * @param data 入力テンソル
 * @param indices インデックスのベクトル (ペアのベクトル)
 * @param updates 更新値のベクトル
 * @return output: 更新されたテンソル
 */
template<typename Derived, typename T>
Eigen::MatrixXd scatternd(const Eigen::MatrixBase<Derived>& data,
                          const std::vector<std::pair<int, int>>& indices,
                          const std::vector<T>& updates) {
    Eigen::MatrixXd output = data;

    for (size_t i = 0; i < indices.size(); ++i) {
        output(indices[i].first, indices[i].second) = updates[i];
    }

    return output;
}

} // namespace onnx

#endif // ONNX_02_SCATTERND_HPP
