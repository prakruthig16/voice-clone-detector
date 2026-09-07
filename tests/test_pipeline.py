"""Smoke tests: feature shape, dataset build, train, and detect all wire up."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.features import extract_features, extract_logmel, feature_names


def _make_clip(tmp_path, name, freq=200):
    import soundfile as sf
    t = np.arange(config.N_SAMPLES) / config.SAMPLE_RATE
    y = 0.5 * np.sin(2 * np.pi * freq * t) + 0.01 * np.random.randn(len(t))
    p = tmp_path / name
    sf.write(p, y, config.SAMPLE_RATE)
    return str(p)


def test_feature_vector_is_finite_and_fixed_length(tmp_path):
    clip = _make_clip(tmp_path, "a.wav")
    v1 = extract_features(clip)
    v2 = extract_features(_make_clip(tmp_path, "b.wav", freq=150))
    assert v1.ndim == 1
    assert v1.shape == v2.shape          # fixed length regardless of content
    assert np.all(np.isfinite(v1))
    assert len(feature_names()) == v1.shape[0]


def test_logmel_shape(tmp_path):
    clip = _make_clip(tmp_path, "c.wav")
    m = extract_logmel(clip, n_mels=64)
    assert m.shape[0] == 64
    assert np.all(np.isfinite(m))


def test_end_to_end_train_and_predict(tmp_path):
    """Tiny synthetic dataset -> train -> predict, using a temp data dir."""
    import soundfile as sf
    from sklearn.model_selection import train_test_split
    from src.models import build_model

    # Two clearly different classes so a model can separate them.
    X, y = [], []
    for i in range(12):
        real = _make_clip(tmp_path, f"r{i}.wav", freq=180)
        fake = _make_clip(tmp_path, f"f{i}.wav", freq=181)  # near-identical tone
        X.append(extract_features(real)); y.append(0)
        X.append(extract_features(fake)); y.append(1)
    X = np.vstack(X); y = np.array(y)

    model = build_model("rf")
    model.fit(X, y)
    proba = model.predict_proba(X[:1])
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
