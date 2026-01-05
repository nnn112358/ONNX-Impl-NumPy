# ONNX オペレータ一覧

## 1. 数学演算 (Math Operations)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Add | 2つのテンソルの要素ごとの加算 | [01_add.py](numpy/01_add.py) | [01_add.hpp](cpp/01_add.hpp) |
| Div | 2つのテンソルの要素ごとの除算 | [01_div.py](numpy/01_div.py) | [01_div.hpp](cpp/01_div.hpp) |
| Mul | 2つのテンソルの要素ごとの乗算 | [01_mul.py](numpy/01_mul.py) | [01_mul.hpp](cpp/01_mul.hpp) |
| Neg | テンソルの符号を反転 | [01_neg.py](numpy/01_neg.py) | [01_neg.hpp](cpp/01_neg.hpp) |
| Pow | テンソルのべき乗演算 | [01_pow.py](numpy/01_pow.py) | [01_pow.hpp](cpp/01_pow.hpp) |
| Sub | 2つのテンソルの要素ごとの減算 | [01_sub.py](numpy/01_sub.py) | [01_sub.hpp](cpp/01_sub.hpp) |
| Exp | テンソルの指数関数 (e^x) | [01_exp.py](numpy/01_exp.py) | [01_exp.hpp](cpp/01_exp.hpp) |
| Log | テンソルの自然対数 | [01_log.py](numpy/01_log.py) | [01_log.hpp](cpp/01_log.hpp) |
| Sqrt | テンソルの平方根 | [01_sqrt.py](numpy/01_sqrt.py) | [01_sqrt.hpp](cpp/01_sqrt.hpp) |
| Clip | テンソルの値を指定範囲にクリップ | [01_clip.py](numpy/01_clip.py) | [01_clip.hpp](cpp/01_clip.hpp) |

## 2. テンソル操作 (Tensor Operations)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Reshape | テンソルの形状を変更 | [02_reshape.py](numpy/02_reshape.py) | [02_reshape.hpp](cpp/02_reshape.hpp) |
| Transpose | テンソルの次元を入れ替え | [02_transpose.py](numpy/02_transpose.py) | [02_transpose.hpp](cpp/02_transpose.hpp) |
| Flatten | テンソルを2次元に平坦化 | [02_flatten.py](numpy/02_flatten.py) | [02_flatten.hpp](cpp/02_flatten.hpp) |
| Squeeze | サイズ1の次元を削除 | [02_squeeze.py](numpy/02_squeeze.py) | [02_squeeze.hpp](cpp/02_squeeze.hpp) |
| Unsqueeze | サイズ1の次元を追加 | [02_unsqueeze.py](numpy/02_unsqueeze.py) | [02_unsqueeze.hpp](cpp/02_unsqueeze.hpp) |
| Resize | テンソルをリサイズ（拡大・縮小） | [02_resize.py](numpy/02_resize.py) | [02_resize.hpp](cpp/02_resize.hpp) |
| Concat | 複数のテンソルを連結 | [02_concat.py](numpy/02_concat.py) | [02_concat.hpp](cpp/02_concat.hpp) |
| Split | テンソルを分割 | [02_split.py](numpy/02_split.py) | [02_split.hpp](cpp/02_split.hpp) |
| Slice | テンソルの一部を切り出し | [02_slice.py](numpy/02_slice.py) | [02_slice.hpp](cpp/02_slice.hpp) |
| Gather | インデックスで要素を収集 | [02_gather.py](numpy/02_gather.py) | [02_gather.hpp](cpp/02_gather.hpp) |
| ScatterND | インデックス位置に値を散布 | [02_scatternd.py](numpy/02_scatternd.py) | [02_scatternd.hpp](cpp/02_scatternd.hpp) |

## 3. ニューラルネットワーク層 (Neural Network Layers)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Conv | 畳み込み演算 | [03_conv.py](numpy/03_conv.py) | [03_conv.hpp](cpp/03_conv.hpp) |
| ConvTranspose | 転置畳み込み（逆畳み込み） | [03_convtranspose.py](numpy/03_convtranspose.py) | [03_convtranspose.hpp](cpp/03_convtranspose.hpp) |
| MaxPool | 最大値プーリング | [03_maxpool.py](numpy/03_maxpool.py) | [03_maxpool.hpp](cpp/03_maxpool.hpp) |
| AveragePool | 平均値プーリング | [03_averagepool.py](numpy/03_averagepool.py) | [03_averagepool.hpp](cpp/03_averagepool.hpp) |
| GlobalAveragePool | グローバル平均プーリング | [03_globalaveragepool.py](numpy/03_globalaveragepool.py) | [03_globalaveragepool.hpp](cpp/03_globalaveragepool.hpp) |
| LayerNormalization | レイヤー正規化 | [03_layernormalization.py](numpy/03_layernormalization.py) | [03_layernormalization.hpp](cpp/03_layernormalization.hpp) |
| LSTM | 長短期記憶ネットワーク | [03_lstm.py](numpy/03_lstm.py) | [03_lstm.hpp](cpp/03_lstm.hpp) |
| GRU | ゲート付き回帰ユニット | [03_gru.py](numpy/03_gru.py) | [03_gru.hpp](cpp/03_gru.hpp) |

## 4. 活性化関数 (Activation Functions)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Relu | ReLU活性化関数 (max(0, x)) | [04_relu.py](numpy/04_relu.py) | [04_relu.hpp](cpp/04_relu.hpp) |
| LeakyRelu | Leaky ReLU活性化関数 | [04_leakyrelu.py](numpy/04_leakyrelu.py) | [04_leakyrelu.hpp](cpp/04_leakyrelu.hpp) |
| Elu | ELU活性化関数 | [04_elu.py](numpy/04_elu.py) | [04_elu.hpp](cpp/04_elu.hpp) |
| PRelu | Parametric ReLU活性化関数 | [04_prelu.py](numpy/04_prelu.py) | [04_prelu.hpp](cpp/04_prelu.hpp) |
| Swish | Swish活性化関数 (x * sigmoid(x)) | [04_swish.py](numpy/04_swish.py) | [04_swish.hpp](cpp/04_swish.hpp) |
| Softmax | Softmax関数（確率分布化） | [04_softmax.py](numpy/04_softmax.py) | [04_softmax.hpp](cpp/04_softmax.hpp) |
| Sigmoid | シグモイド関数 (1/(1+e^-x)) | [04_sigmoid.py](numpy/04_sigmoid.py) | [04_sigmoid.hpp](cpp/04_sigmoid.hpp) |
| HardSigmoid | Hard Sigmoid（区分線形近似） | [04_hardsigmoid.py](numpy/04_hardsigmoid.py) | [04_hardsigmoid.hpp](cpp/04_hardsigmoid.hpp) |
| HardSwish | Hard Swish活性化関数 | [04_hardswish.py](numpy/04_hardswish.py) | [04_hardswish.hpp](cpp/04_hardswish.hpp) |
| Tanh | ハイパボリックタンジェント | [04_tanh.py](numpy/04_tanh.py) | [04_tanh.hpp](cpp/04_tanh.hpp) |

## 5. 線形代数 (Linear Algebra)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| MatMul | 行列乗算 | [05_matmul.py](numpy/05_matmul.py) | [05_matmul.hpp](cpp/05_matmul.hpp) |
| Gemm | 一般行列乗算 (alpha*A*B + beta*C) | [05_gemm.py](numpy/05_gemm.py) | [05_gemm.hpp](cpp/05_gemm.hpp) |

## 6. 比較演算 (Comparison Operations)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Equal | 要素ごとの等価比較 (A == B) | [06_equal.py](numpy/06_equal.py) | [06_equal.hpp](cpp/06_equal.hpp) |
| Greater | 要素ごとの大なり比較 (A > B) | [06_greater.py](numpy/06_greater.py) | [06_greater.hpp](cpp/06_greater.hpp) |
| GreaterOrEqual | 要素ごとの以上比較 (A >= B) | [06_greaterorequal.py](numpy/06_greaterorequal.py) | [06_greaterorequal.hpp](cpp/06_greaterorequal.hpp) |
| Less | 要素ごとの小なり比較 (A < B) | [06_less.py](numpy/06_less.py) | [06_less.hpp](cpp/06_less.hpp) |
| LessOrEqual | 要素ごとの以下比較 (A <= B) | [06_lessorequal.py](numpy/06_lessorequal.py) | [06_lessorequal.hpp](cpp/06_lessorequal.hpp) |

## 7. 集約・統計演算 (Reduction Operations)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| ReduceSum | 指定軸に沿った合計 | [07_reducesum.py](numpy/07_reducesum.py) | [07_reducesum.hpp](cpp/07_reducesum.hpp) |
| ReduceMean | 指定軸に沿った平均 | [07_reducemean.py](numpy/07_reducemean.py) | [07_reducemean.hpp](cpp/07_reducemean.hpp) |
| ReduceMax | 指定軸に沿った最大値 | [07_reducemax.py](numpy/07_reducemax.py) | [07_reducemax.hpp](cpp/07_reducemax.hpp) |
| ReduceMin | 指定軸に沿った最小値 | [07_reducemin.py](numpy/07_reducemin.py) | [07_reducemin.hpp](cpp/07_reducemin.hpp) |
| ReduceProd | 指定軸に沿った積 | [07_reduceprod.py](numpy/07_reduceprod.py) | [07_reduceprod.hpp](cpp/07_reduceprod.hpp) |
| ReduceL2 | L2ノルム (√Σx²) | [07_reducel2.py](numpy/07_reducel2.py) | [07_reducel2.hpp](cpp/07_reducel2.hpp) |
| ReduceL1 | L1ノルム (Σ\|x\|) | [07_reducel1.py](numpy/07_reducel1.py) | [07_reducel1.hpp](cpp/07_reducel1.hpp) |
| ReduceSumSquare | 二乗和 (Σx²) | [07_reducesumsquare.py](numpy/07_reducesumsquare.py) | [07_reducesumsquare.hpp](cpp/07_reducesumsquare.hpp) |
| ReduceLogSumExp | log(Σe^x)（数値安定版） | [07_reducelogsumexp.py](numpy/07_reducelogsumexp.py) | [07_reducelogsumexp.hpp](cpp/07_reducelogsumexp.hpp) |
| ReduceLogSum | log(Σx) | [07_reducelogsum.py](numpy/07_reducelogsum.py) | [07_reducelogsum.hpp](cpp/07_reducelogsum.hpp) |

## 8. ユーティリティ (Utility Operations)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| Pad | テンソルにパディングを追加 | [08_pad.py](numpy/08_pad.py) | [08_pad.hpp](cpp/08_pad.hpp) |

## 9. 画像処理 (Image Processing)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| SpaceToDepth | 空間次元をチャネル次元に変換 | [09_spacetodepth.py](numpy/09_spacetodepth.py) | [09_spacetodepth.hpp](cpp/09_spacetodepth.hpp) |
| DepthToSpace | チャネル次元を空間次元に変換 | [09_depthtospace.py](numpy/09_depthtospace.py) | [09_depthtospace.hpp](cpp/09_depthtospace.hpp) |

## 10. 制御フロー (Control Flow)

| オペレータ | 説明 | Python 実装 | C++ 実装 |
|-----------|------|------------|---------|
| ReverseSequence | シーケンスを指定長まで反転 | [10_reversesequence.py](numpy/10_reversesequence.py) | [10_reversesequence.hpp](cpp/10_reversesequence.hpp) |

---

**合計**: 60オペレータ

## 📚 参考

各オペレータの詳細な仕様については、[ONNX公式ドキュメント](https://onnx.ai/onnx/operators/)を参照してください。
