# sales-prediction
# 売上予測システム（Sales Prediction System）

## ■ 概要
過去の売上データから、将来の売上を予測するシステムです。  
Pandasによるデータ分析と機械学習を用いて実装しました。

---

## ■ 使用技術
- Python
- Pandas
- scikit-learn
- matplotlib

---

## ■ 機能
- 売上データの読み込み・加工
- 日別売上の集計
- 特徴量作成（曜日・ラグ・移動平均）
- 売上予測（RandomForest）
- 精度評価（RMSE）
- グラフによる可視化
- 未来の売上予測

---

## ■ 工夫した点（ここ重要）
- ラグ特徴量を追加し、時系列の影響を考慮
- 曜日・月を追加し、季節性を反映
- モデルを線形回帰からRandomForestに変更し精度改善
- 時系列データのため、shuffle=Falseで分割

---

## ■ 実行方法

### 1. 環境構築
```bash
pip install pandas scikit-learn matplotlib
