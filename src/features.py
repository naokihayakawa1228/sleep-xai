# -*- coding: utf-8 -*-
"""
src/features.py
================
共通の特徴量抽出処理をまとめるモジュール。

【変更履歴】
- Step1: eeg_power_band() を MNE tutorialからそのまま移動 (中身は無変更)
- Step2: extract_permana_features() を新規追加
         (Permana et al. 2025 スタイルの22特徴量)
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
import pywt
import antropy as ant

# ---------------------------------------------------------------------
# Step1: MNE tutorialのeeg_power_band関数 (中身は一切変更していません)
# ---------------------------------------------------------------------
FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30],
}


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    tutorialのコードそのまま (変更なし)。
    """
    spectrum = epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)
    psds, freqs = spectrum.get_data(return_freqs=True)
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


# ---------------------------------------------------------------------
# Step2: 新規追加 (Permana et al. 2025 スタイルの22特徴量)
# ---------------------------------------------------------------------
#
# eeg_power_bandはPSD(周波数領域)から帯域パワーだけを計算していたが、
# Permanaの特徴量セットには「生波形(時間領域)」から計算するもの
# (統計量・エントロピー・spindle/K-complex数・wavelet)が多く含まれるため、
# エポックごとに生波形を取り出してループ処理する新しい関数を追加する。
#
# 22特徴量の内訳:
#   統計量(4)      : mean, variance, skewness, kurtosis
#   エントロピー(3) : shannon entropy, permutation entropy, sample entropy
#   波形カウント(2) : spindle count, K-complex count (簡易検出)
#   帯域比(3)      : theta/alpha, delta/beta, (theta+alpha)/beta
#   帯域パワー(5)   : delta, theta, alpha, sigma, beta (相対パワー)
#   wavelet(5)     : db4で5段階分解した各detail係数のエネルギー

PERMANA_FEATURE_NAMES = [
    "mean", "variance", "skewness", "kurtosis",
    "shannon_entropy", "permutation_entropy", "sample_entropy",
    "spindle_count", "kcomplex_count",
    "ratio_theta_alpha", "ratio_delta_beta", "ratio_thetaalpha_beta",
    "bp_delta", "bp_theta", "bp_alpha", "bp_sigma", "bp_beta",
    "wavelet_energy_d1", "wavelet_energy_d2", "wavelet_energy_d3",
    "wavelet_energy_d4", "wavelet_energy_d5",
]


def _band_relative_power_1d(sig, sfreq):
    """1エポック分の生波形から、Welch法で帯域相対パワーを計算する。"""
    freqs, psd = sp_signal.welch(sig, fs=sfreq, nperseg=min(len(sig), int(sfreq * 4)))
    total_power = np.sum(psd) + 1e-12
    powers = {}
    for name, (fmin, fmax) in FREQ_BANDS.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        powers[name] = np.sum(psd[mask]) / total_power
    return powers


def _shannon_entropy_1d(sig, n_bins=32):
    hist, _ = np.histogram(sig, bins=n_bins, density=True)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return -np.sum(p * np.log2(p))


def _count_spindles(sig, sfreq, low=11.0, high=16.0, min_dur=0.5, max_dur=2.0, thresh_std=1.5):
    """
    簡易スピンドル検出 (振幅包絡ベース)。
    ※ A7アルゴリズム等の厳密な検出法ではなく、教育的な簡易版。
    """
    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = sp_signal.sosfiltfilt(sos, sig)
    envelope = np.abs(sp_signal.hilbert(filtered))
    thresh = envelope.mean() + thresh_std * envelope.std()
    above = envelope > thresh

    count, dur = 0, 0
    min_samp, max_samp = min_dur * sfreq, max_dur * sfreq
    for val in above:
        if val:
            dur += 1
        else:
            if min_samp <= dur <= max_samp:
                count += 1
            dur = 0
    if min_samp <= dur <= max_samp:
        count += 1
    return count


def _count_kcomplexes(sig, sfreq, low=0.5, high=4.0, min_dur=0.5, max_dur=1.5, thresh_percentile=90):
    """
    簡易K-complex検出 (デルタ帯域の大振幅偏位ベース)。
    ※ 厳密な検出アルゴリズムではなく、教育的な簡易版。
    """
    sos = sp_signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = sp_signal.sosfiltfilt(sos, sig)
    thresh = np.percentile(np.abs(filtered), thresh_percentile)
    above = np.abs(filtered) > thresh

    count, dur = 0, 0
    min_samp, max_samp = min_dur * sfreq, max_dur * sfreq
    for val in above:
        if val:
            dur += 1
        else:
            if min_samp <= dur <= max_samp:
                count += 1
            dur = 0
    if min_samp <= dur <= max_samp:
        count += 1
    return count


def _wavelet_energies(sig, wavelet="db4", level=5):
    coeffs = pywt.wavedec(sig, wavelet=wavelet, level=level)
    details = coeffs[1:]  # coeffs[0]はapproximation
    energies = [np.sum(np.square(c)) for c in details]
    energies = energies[::-1]  # level(粗い)→1(細かい)の順を d1..d5 に揃える
    energies = (energies + [0.0] * 5)[:5]
    return energies


def _extract_permana_features_1epoch(sig, sfreq):
    """1エポック(1次元の生波形)から22特徴量を計算する。"""
    feat = []

    # 統計量(4)
    feat += [np.mean(sig), np.var(sig), skew(sig), kurtosis(sig)]

    # エントロピー(3) ← antropyを使用
    feat.append(_shannon_entropy_1d(sig))
    feat.append(ant.perm_entropy(sig, order=3, normalize=True))
    feat.append(ant.sample_entropy(sig))

    # spindle / K-complex count(2)
    feat.append(_count_spindles(sig, sfreq))
    feat.append(_count_kcomplexes(sig, sfreq))

    # 帯域比(3)
    bp = _band_relative_power_1d(sig, sfreq)
    eps = 1e-12
    feat.append(bp["theta"] / (bp["alpha"] + eps))
    feat.append(bp["delta"] / (bp["beta"] + eps))
    feat.append((bp["theta"] + bp["alpha"]) / (bp["beta"] + eps))

    # 帯域パワー(5)
    feat += [bp["delta"], bp["theta"], bp["alpha"], bp["sigma"], bp["beta"]]

    # wavelet係数エネルギー(5) ← pywtを使用
    feat += _wavelet_energies(sig)

    return np.array(feat, dtype=float)


def extract_permana_features(epochs, verbose=False):
    """
    mne.Epochs を受け取り、Permana 2025スタイルの22特徴量を返す。
    eeg_power_bandと同じインターフェース(Epochs -> ndarray)にしてあるので、
    Step1のパイプラインの中で eeg_power_band と入れ替えるだけで使える。

    Parameters
    ----------
    epochs : mne.Epochs

    Returns
    -------
    X : ndarray, shape (n_epochs, 22)
    """
    data = epochs.get_data(picks="eeg")  # shape: (n_epochs, 1, n_samples)
    sfreq = epochs.info["sfreq"]
    n_epochs = data.shape[0]

    X = np.zeros((n_epochs, 22))
    for i in range(n_epochs):
        X[i] = _extract_permana_features_1epoch(data[i, 0, :], sfreq)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_epochs} エポック完了")
    return X
