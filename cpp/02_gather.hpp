#ifndef ONNX_02_GATHER_HPP
#define ONNX_02_GATHER_HPP

#include <Eigen/Dense>
#include <vector>

namespace onnx {

/**
 * ONNX Gather operator
 *
 * 指定された軸に沿って、インデックスで指定された要素を収集する。
 *
 * @param data 入力テンソル
 * @param indices 収集するインデックスのベクトル
 * @param axis 収集する軸 (0: 行, 1: 列)
 * @return output: 収集された要素
 */
template<typename Derived>
Eigen::MatrixXd gather(const Eigen::MatrixBase<Derived>& data,
                       const std::vector<int>& indices,
                       int axis = 0) {
    if (axis == 0) {
        // 行方向に収集
        Eigen::MatrixXd result(indices.size(), data.cols());

        for (size_t i = 0; i < indices.size(); ++i) {
            result.row(i) = data.row(indices[i]);
        }

        return result;
    } else {
        // 列方向に収集
        Eigen::MatrixXd result(data.rows(), indices.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            result.col(i) = data.col(indices[i]);
        }

        return result;
    }
}

/**
 * Gather with Eigen vector indices
 *
 * @param data 入力テンソル
 * @param indices 収集するインデックスのEigenベクトル
 * @param axis 収集する軸 (0: 行, 1: 列)
 * @return output: 収集された要素
 */
template<typename Derived1, typename Derived2>
Eigen::MatrixXd gather(const Eigen::MatrixBase<Derived1>& data,
                       const Eigen::MatrixBase<Derived2>& indices,
                       int axis = 0) {
    std::vector<int> idx_vec;
    for (int i = 0; i < indices.size(); ++i) {
        idx_vec.push_back(static_cast<int>(indices(i)));
    }

    return gather(data, idx_vec, axis);
}

} // namespace onnx

#endif // ONNX_02_GATHER_HPP
