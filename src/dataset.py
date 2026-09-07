"""
Dataset assembly: walk the data/real and data/fake folders, extract a feature
vector per clip, and return an (X, y) matrix ready for scikit-learn.

Features are cached to models/features.npz keyed by a hash of the file list so
repeated training runs don't re-extract everything.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

import config
from src.features import extract_features

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _list_audio(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _fingerprint(files: list[Path]) -> str:
    h = hashlib.sha1()
    for f in files:
        stat = f.stat()
        h.update(str(f).encode())
        h.update(str(stat.st_mtime_ns).encode())
        h.update(str(stat.st_size).encode())
    return h.hexdigest()


def build_dataset(use_cache: bool = True, verbose: bool = True):
    """Return (X, y, paths). X: (n, d) float32, y: (n,) int labels."""
    real = _list_audio(config.REAL_DIR)
    fake = _list_audio(config.FAKE_DIR)
    if not real or not fake:
        raise RuntimeError(
            f"Need audio in both {config.REAL_DIR} and {config.FAKE_DIR}. "
            f"Found {len(real)} real, {len(fake)} fake. "
            "Run scripts/generate_demo_data.py or add your own dataset."
        )

    files = real + fake
    labels = [config.LABEL_REAL] * len(real) + [config.LABEL_FAKE] * len(fake)
    fp = _fingerprint(files)

    if use_cache and config.FEATURE_CACHE.exists():
        cached = np.load(config.FEATURE_CACHE, allow_pickle=True)
        if str(cached.get("fingerprint")) == fp:
            if verbose:
                print(f"Loaded cached features: {cached['X'].shape}")
            return cached["X"], cached["y"], list(cached["paths"])

    feats, kept_labels, kept_paths = [], [], []
    it = tqdm(list(zip(files, labels)), desc="Extracting features") if verbose \
        else zip(files, labels)
    for path, label in it:
        try:
            feats.append(extract_features(str(path)))
            kept_labels.append(label)
            kept_paths.append(str(path))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipping {path.name}: {e}")

    if not feats:
        raise RuntimeError("Feature extraction failed for every clip.")

    X = np.vstack(feats)
    y = np.asarray(kept_labels, dtype=np.int64)
    paths = kept_paths

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.FEATURE_CACHE, X=X, y=y,
                        paths=np.array(paths), fingerprint=fp)
    if verbose:
        print(f"Extracted features: X={X.shape}, real={np.sum(y==0)}, "
              f"fake={np.sum(y==1)}")
    return X, y, paths
