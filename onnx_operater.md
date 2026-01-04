# ONNX オペレータ一覧

## 1. 数学演算 (Math Operations)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Add | 2つのテンソルの要素ごとの加算 | [01_add.py](numpy/01_add.py) |
| Div | 2つのテンソルの要素ごとの除算 | [01_div.py](numpy/01_div.py) |
| Mul | 2つのテンソルの要素ごとの乗算 | [01_mul.py](numpy/01_mul.py) |
| Neg | テンソルの符号を反転 | [01_neg.py](numpy/01_neg.py) |
| Pow | テンソルのべき乗演算 | [01_pow.py](numpy/01_pow.py) |
| Sub | 2つのテンソルの要素ごとの減算 | [01_sub.py](numpy/01_sub.py) |
| Exp | テンソルの指数関数 (e^x) | [01_exp.py](numpy/01_exp.py) |
| Log | テンソルの自然対数 | [01_log.py](numpy/01_log.py) |
| Sqrt | テンソルの平方根 | [01_sqrt.py](numpy/01_sqrt.py) |
| Clip | テンソルの値を指定範囲にクリップ | [01_clip.py](numpy/01_clip.py) |

## 2. テンソル操作 (Tensor Operations)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Reshape | テンソルの形状を変更 | [02_reshape.py](numpy/02_reshape.py) |
| Transpose | テンソルの次元を入れ替え | [02_transpose.py](numpy/02_transpose.py) |
| Flatten | テンソルを2次元に平坦化 | [02_flatten.py](numpy/02_flatten.py) |
| Squeeze | サイズ1の次元を削除 | [02_squeeze.py](numpy/02_squeeze.py) |
| Unsqueeze | サイズ1の次元を追加 | [02_unsqueeze.py](numpy/02_unsqueeze.py) |
| Resize | テンソルをリサイズ（拡大・縮小） | [02_resize.py](numpy/02_resize.py) |
| Concat | 複数のテンソルを連結 | [02_concat.py](numpy/02_concat.py) |
| Split | テンソルを分割 | [02_split.py](numpy/02_split.py) |
| Slice | テンソルの一部を切り出し | [02_slice.py](numpy/02_slice.py) |
| Gather | インデックスで要素を収集 | [02_gather.py](numpy/02_gather.py) |
| ScatterND | インデックス位置に値を散布 | [02_scatternd.py](numpy/02_scatternd.py) |

## 3. ニューラルネットワーク層 (Neural Network Layers)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Conv | 畳み込み演算 | [03_conv.py](numpy/03_conv.py) |
| ConvTranspose | 転置畳み込み（逆畳み込み） | [03_convtranspose.py](numpy/03_convtranspose.py) |
| MaxPool | 最大値プーリング | [03_maxpool.py](numpy/03_maxpool.py) |
| AveragePool | 平均値プーリング | [03_averagepool.py](numpy/03_averagepool.py) |
| GlobalAveragePool | グローバル平均プーリング | [03_globalaveragepool.py](numpy/03_globalaveragepool.py) |
| LayerNormalization | レイヤー正規化 | [03_layernormalization.py](numpy/03_layernormalization.py) |
| LSTM | 長短期記憶ネットワーク | [03_lstm.py](numpy/03_lstm.py) |
| GRU | ゲート付き回帰ユニット | [03_gru.py](numpy/03_gru.py) |

## 4. 活性化関数 (Activation Functions)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Relu | ReLU活性化関数 (max(0, x)) | [04_relu.py](numpy/04_relu.py) |
| LeakyRelu | Leaky ReLU活性化関数 | [04_leakyrelu.py](numpy/04_leakyrelu.py) |
| Elu | ELU活性化関数 | [04_elu.py](numpy/04_elu.py) |
| PRelu | Parametric ReLU活性化関数 | [04_prelu.py](numpy/04_prelu.py) |
| Swish | Swish活性化関数 (x * sigmoid(x)) | [04_swish.py](numpy/04_swish.py) |
| Softmax | Softmax関数（確率分布化） | [04_softmax.py](numpy/04_softmax.py) |
| Sigmoid | シグモイド関数 (1/(1+e^-x)) | [04_sigmoid.py](numpy/04_sigmoid.py) |
| HardSigmoid | Hard Sigmoid（区分線形近似） | [04_hardsigmoid.py](numpy/04_hardsigmoid.py) |
| HardSwish | Hard Swish活性化関数 | [04_hardswish.py](numpy/04_hardswish.py) |
| Tanh | ハイパボリックタンジェント | [04_tanh.py](numpy/04_tanh.py) |

## 5. 線形代数 (Linear Algebra)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| MatMul | 行列乗算 | [05_matmul.py](numpy/05_matmul.py) |
| Gemm | 一般行列乗算 (alpha*A*B + beta*C) | [05_gemm.py](numpy/05_gemm.py) |

## 6. 比較演算 (Comparison Operations)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Equal | 要素ごとの等価比較 (A == B) | [06_equal.py](numpy/06_equal.py) |
| Greater | 要素ごとの大なり比較 (A > B) | [06_greater.py](numpy/06_greater.py) |
| GreaterOrEqual | 要素ごとの以上比較 (A >= B) | [06_greaterorequal.py](numpy/06_greaterorequal.py) |
| Less | 要素ごとの小なり比較 (A < B) | [06_less.py](numpy/06_less.py) |
| LessOrEqual | 要素ごとの以下比較 (A <= B) | [06_lessorequal.py](numpy/06_lessorequal.py) |

## 7. 集約・統計演算 (Reduction Operations)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| ReduceSum | 指定軸に沿った合計 | [07_reducesum.py](numpy/07_reducesum.py) |
| ReduceMean | 指定軸に沿った平均 | [07_reducemean.py](numpy/07_reducemean.py) |
| ReduceMax | 指定軸に沿った最大値 | [07_reducemax.py](numpy/07_reducemax.py) |
| ReduceMin | 指定軸に沿った最小値 | [07_reducemin.py](numpy/07_reducemin.py) |
| ReduceProd | 指定軸に沿った積 | [07_reduceprod.py](numpy/07_reduceprod.py) |
| ReduceL2 | L2ノルム (√Σx²) | [07_reducel2.py](numpy/07_reducel2.py) |
| ReduceL1 | L1ノルム (Σ\|x\|) | [07_reducel1.py](numpy/07_reducel1.py) |
| ReduceSumSquare | 二乗和 (Σx²) | [07_reducesumsquare.py](numpy/07_reducesumsquare.py) |
| ReduceLogSumExp | log(Σe^x)（数値安定版） | [07_reducelogsumexp.py](numpy/07_reducelogsumexp.py) |
| ReduceLogSum | log(Σx) | [07_reducelogsum.py](numpy/07_reducelogsum.py) |

## 8. ユーティリティ (Utility Operations)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| Pad | テンソルにパディングを追加 | [08_pad.py](numpy/08_pad.py) |

## 9. 画像処理 (Image Processing)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| SpaceToDepth | 空間次元をチャネル次元に変換 | [09_spacetodepth.py](numpy/09_spacetodepth.py) |
| DepthToSpace | チャネル次元を空間次元に変換 | [09_depthtospace.py](numpy/09_depthtospace.py) |

## 10. 制御フロー (Control Flow)

| オペレータ | 説明 | 実装ファイル |
|-----------|------|------------|
| ReverseSequence | シーケンスを指定長まで反転 | [10_reversesequence.py](numpy/10_reversesequence.py) |

---

**合計**: 60オペレータ

## 📚 参考

各オペレータの詳細な仕様については、[ONNX公式ドキュメント](https://onnx.ai/onnx/operators/)を参照してください。
