# ONNX-Impl-NumPy

ONNXオペレータをNumPyで実装したリファレンス実装集です。各オペレータはONNX仕様に準拠し、純粋なNumPyで同等の計算を行います。

## 📁 プロジェクト構造

```
ONNX-Impl-NumPy/
├── README.md                    # このファイル
├── onnx_operater.md            # オペレータ一覧（リンク付き）
├── requirements.txt            # 依存パッケージ
└── numpy/                      # 実装ファイル
    ├── 01_*.py                 # 数学演算 (10個)
    ├── 02_*.py                 # テンソル操作 (11個)
    ├── 03_*.py                 # ニューラルネットワーク層 (8個)
    ├── 04_*.py                 # 活性化関数 (10個)
    ├── 05_*.py                 # 線形代数 (2個)
    ├── 06_*.py                 # 比較演算 (5個)
    ├── 07_*.py                 # 集約・統計演算 (10個)
    ├── 08_*.py                 # ユーティリティ (1個)
    ├── 09_*.py                 # 画像処理 (2個)
    └── 10_*.py                 # 制御フロー (1個)
```

## 📋 実装オペレータ一覧

全60個のONNXオペレータを実装しています。詳細は [onnx_operater.md](onnx_operater.md) を参照してください。

### 1. 数学演算 (10個)
Add, Div, Mul, Neg, Pow, Sub, Exp, Log, Sqrt, Clip

### 2. テンソル操作 (11個)
Reshape, Transpose, Flatten, Squeeze, Unsqueeze, Resize, Concat, Split, Slice, Gather, ScatterND

### 3. ニューラルネットワーク層 (8個)
Conv, ConvTranspose, MaxPool, AveragePool, GlobalAveragePool, LayerNormalization, LSTM, GRU

### 4. 活性化関数 (10個)
Relu, LeakyRelu, Elu, PRelu, Swish, Softmax, Sigmoid, HardSigmoid, HardSwish, Tanh

### 5. 線形代数 (2個)
MatMul, Gemm

### 6. 比較演算 (5個)
Equal, Greater, GreaterOrEqual, Less, LessOrEqual

### 7. 集約・統計演算 (10個)
ReduceSum, ReduceMean, ReduceMax, ReduceMin, ReduceProd, ReduceL2, ReduceL1, ReduceSumSquare, ReduceLogSumExp, ReduceLogSum

### 8. ユーティリティ (1個)
Pad

### 9. 画像処理 (2個)
SpaceToDepth, DepthToSpace

### 10. 制御フロー (1個)
ReverseSequence

## 🚀 使い方

各ファイルは独立して実行可能で、関数として利用できます。

### 基本的な使用例

```python
# Add オペレータの例
from numpy.01_add import add
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = add(A, B)
print(C)
# 出力:
# [[ 6  8]
#  [10 12]]
```

### テストコードの実行

各ファイルには`if __name__ == "__main__"`ブロックにテストコードが含まれています。

```bash
# 個別のオペレータをテスト
python numpy/01_add.py

# 複数のオペレータをテスト
python numpy/04_relu.py
python numpy/03_conv.py
```

## 📦 必要な依存関係

- Python 3.8+
- NumPy 1.19+
- SciPy (Resizeオペレータで使用)

### インストール

[uv](https://github.com/astral-sh/uv)を使用した高速インストール（推奨）：

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# requirements.txtからインストール
uv pip install -r requirements.txt

# または直接指定
uv pip install numpy scipy
```

従来のpipを使用する場合：

```bash
pip install -r requirements.txt
```

## 💡 各ファイルの構成

すべての実装ファイルは以下の構造を持っています：

1. **関数定義**: オペレータの実装
2. **Docstring**: 機能説明、パラメータ、戻り値の詳細
3. **テストコード**: `if __name__ == "__main__"` ブロック内の実行例

### ファイル例

```python
import numpy as np

def add(A, B):
    """
    ONNX Add オペレータ

    2つのテンソルの要素ごとの加算を行う。
    ブロードキャストをサポート。

    Args:
        A: 入力テンソル1
        B: 入力テンソル2

    Returns:
        C: A + B の結果
    """
    return np.add(A, B)

if __name__ == "__main__":
    # テスト例
    ...
```

## 📝 ファイル命名規則

ファイル名は以下の形式で統一されています：

```
{カテゴリ番号:02d}_{オペレータ名小文字}.py
```

例：
- `01_add.py` - 数学演算カテゴリのAddオペレータ
- `04_relu.py` - 活性化関数カテゴリのReluオペレータ
- `07_reducesum.py` - 集約・統計演算カテゴリのReduceSumオペレータ

## 🔍 オペレータの詳細

各オペレータの詳細な仕様については、[ONNX公式ドキュメント](https://onnx.ai/onnx/operators/)を参照してください。

## ⚠️ 注意事項

- この実装はリファレンス実装であり、最適化されていません
- 本番環境での使用には、ONNX RuntimeやPyTorchなどの最適化されたライブラリを推奨します
- 一部のオペレータは簡略化された実装となっています
- 数値精度や境界条件については、実際のONNX Runtimeと異なる場合があります

## 📚 参考資料

- [ONNX公式サイト](https://onnx.ai/)
- [ONNX Operators Documentation](https://onnx.ai/onnx/operators/)
- [NumPy Documentation](https://numpy.org/doc/)

## 🤝 コントリビューション

バグ報告や改善提案は歓迎します。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**作成日**: 2026年1月4日
**実装数**: 60オペレータ
