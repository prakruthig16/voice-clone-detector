"""
Feature extraction for voice-clone / impersonation detection.

The intuition: text-to-speech and voice-conversion (cloning) systems leave
subtle acoustic artefacts that differ from genuine human speech. We capture
these with four families of features:

  1. Spectral / cepstral  -> MFCCs (+ deltas), spectral centroid, bandwidth,
     rolloff, flatness, zero-crossing rate. Vocoders smooth fine spectral
     detail and often over-regularise the higher cepstral coefficients.

  2. Prosodic             -> pitch (F0) statistics, voiced ratio. Synthetic
     voices tend to have flatter, more regular intonation.

  3. Voice-quality        -> spectral contrast, harmonic-to-noise proxy,
     RMS energy dynamics. Cloned speech is often "too clean".

  4. Phase / artefact     -> spectral flatness of the residual and the
     variance of the log-mel deltas, which pick up vocoder buzz.

Every clip is reduced to a single fixed-length vector by taking the mean and
standard deviation over time of each frame-level feature. That vector feeds a
classical ML classifier (see models.py). The same functions can emit a 2-D
log-mel spectrogram for the optional CNN backend.
"""
from __future__ import annotations

import warnings

import numpy as np
import librosa

import config

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Loading & normalisation
# --------------------------------------------------------------------------
def load_audio(path: str, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load a mono clip at the target sample rate, trimmed/padded to DURATION."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = _fix_length(y, config.N_SAMPLES)
    # Peak-normalise so loudness differences don't leak into the classifier.
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def _fix_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y


def _stats(x: np.ndarray) -> list[float]:
    """Mean + std summary of a frame-level feature vector or matrix (axis=time)."""
    x = np.atleast_2d(x)
    return np.concatenate([x.mean(axis=1), x.std(axis=1)]).tolist()


# --------------------------------------------------------------------------
# Frame-level feature vector (for classical models)
# --------------------------------------------------------------------------
def extract_features(path: str) -> np.ndarray:
    """Return a 1-D feature vector describing one audio clip."""
    y = load_audio(path)
    sr = config.SAMPLE_RATE
    feats: list[float] = []

    # 1. MFCCs + deltas -------------------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats += _stats(mfcc)
    feats += _stats(d1)
    feats += _stats(d2)

    # 2. Spectral shape -------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH))
    feats += _stats(librosa.feature.spectral_centroid(S=S, sr=sr))
    feats += _stats(librosa.feature.spectral_bandwidth(S=S, sr=sr))
    feats += _stats(librosa.feature.spectral_rolloff(S=S, sr=sr))
    feats += _stats(librosa.feature.spectral_flatness(S=S))
    feats += _stats(librosa.feature.spectral_contrast(S=S, sr=sr))
    feats += _stats(librosa.feature.zero_crossing_rate(y,
                    frame_length=config.WIN_LENGTH, hop_length=config.HOP_LENGTH))

    # 3. Energy dynamics -----------------------------------------------
    rms = librosa.feature.rms(y=y, hop_length=config.HOP_LENGTH)
    feats += _stats(rms)

    # 4. Prosody: pitch (F0) -------------------------------------------
    f0, voiced_flag, _ = librosa.pyin(
        y, sr=sr, fmin=65, fmax=400,
        frame_length=config.N_FFT, hop_length=config.HOP_LENGTH,
    )
    f0_valid = f0[~np.isnan(f0)]
    if f0_valid.size:
        feats += [float(np.mean(f0_valid)), float(np.std(f0_valid)),
                  float(np.min(f0_valid)), float(np.max(f0_valid))]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    feats.append(float(np.mean(voiced_flag)))  # voiced ratio

    # 5. Vocoder-artefact proxy: variance of log-mel deltas ------------
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-10)
    feats.append(float(np.var(librosa.feature.delta(logmel))))

    return np.asarray(feats, dtype=np.float32)


def feature_names() -> list[str]:
    """Human-readable names aligned with extract_features output (best-effort)."""
    names = []
    for group, n in [("mfcc", config.N_MFCC), ("mfcc_d1", config.N_MFCC),
                     ("mfcc_d2", config.N_MFCC)]:
        names += [f"{group}{i}_mean" for i in range(n)]
        names += [f"{group}{i}_std" for i in range(n)]
    for group, n in [("centroid", 1), ("bandwidth", 1), ("rolloff", 1),
                     ("flatness", 1), ("contrast", 7), ("zcr", 1), ("rms", 1)]:
        names += [f"{group}{i}_mean" for i in range(n)]
        names += [f"{group}{i}_std" for i in range(n)]
    names += ["f0_mean", "f0_std", "f0_min", "f0_max", "voiced_ratio",
              "logmel_delta_var"]
    return names


# --------------------------------------------------------------------------
# 2-D spectrogram (for the optional CNN backend)
# --------------------------------------------------------------------------
def extract_logmel(path: str, n_mels: int = 64) -> np.ndarray:
    """Return a (n_mels, T) normalised log-mel spectrogram."""
    y = load_audio(path)
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SAMPLE_RATE, n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH, n_mels=n_mels)
    logmel = librosa.power_to_db(mel + 1e-10)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)
