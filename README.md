# Sleep XAI Project

EEGを用いた睡眠段階分類に対して、特徴量抽出・機械学習・SHAPによるXAIを段階的に実装するプロジェクト。

## Goal

- Sleep-EDFを用いた睡眠段階分類
- MNE tutorialをbaselineとして再現
- 特徴量を拡張
- RandomForest / XGBoostで分類
- SHAPで特徴量重要度を可視化
- 睡眠医学的に結果を解釈

## Structure

- notebooks/: ColabやJupyterで実行する実験用Notebook
- src/: 共通処理
- outputs/: 数値結果
- figures/: 図
- models/: 学習済みモデル
- results/: 実験ごとの結果

## Steps

1. Baseline: MNE tutorial
2. Feature Extension: Permana-style features
3. Model Comparison: RandomForest vs XGBoost
4. XAI: SHAP analysis
