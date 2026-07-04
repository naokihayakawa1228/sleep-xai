# -*- coding: utf-8 -*-
"""
src/shap_utils.py
==================
shapのバージョン・モデルによってTreeExplainer.shap_values()の
返り値の形状が異なる問題を吸収するヘルパー。
(RandomForestClassifierの場合、通常 (n_samples, n_features, n_classes) で返る)
"""

import numpy as np


def normalize_shap_values(shap_values, n_classes, n_features):
    """常に (n_classes, n_samples, n_features) の形状に正規化する。"""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        shape = arr.shape
        if shape[0] == n_classes and shape[2] == n_features:
            return arr
        if shape[-1] == n_classes and shape[1] == n_features:
            return np.transpose(arr, (2, 0, 1))
        if shape[1] == n_classes and shape[2] == n_features:
            return np.transpose(arr, (1, 0, 2))

    raise ValueError(f"想定外のSHAP出力形状です: {arr.shape}")