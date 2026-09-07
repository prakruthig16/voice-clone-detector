"""
Generate synthetic demo audio so the pipeline runs end-to-end WITHOUT a real
dataset. This is for wiring/testing only -- it is NOT a substitute for a proper
corpus like ASVspoof. See the README for real datasets.

Design of the two classes (deliberately separable but non-trivial):

  REAL  -> a source-filter model of voiced speech: a glottal pulse train with
           natural jitter/shimmer, passed through 3-4 randomly drifting formant
           resonances, plus breath noise. Pitch contour wanders naturally.

  FAKE  -> a "vocoder-like" rendition: same rough pitch but with an overly
           regular pulse train, quantised/smoothed formants, reduced jitter,
           and a faint high-frequency buzz -- mimicking the artefacts real
           clone detectors key on.

Usage:
    python scripts/generate_demo_data.py --n 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SR = config.SAMPLE_RATE
DUR = config.DURATION


def _formant_filter(x, freqs, sr, bw=90.0):
    """Apply a cascade of 2nd-order resonators (simple formant model)."""
    y = x.copy()
    r = np.exp(-np.pi * bw / sr)
    for f in freqs:
        theta = 2 * np.pi * f / sr
        b = [1 - r]
        a = [1.0, -2 * r * np.cos(theta), r * r]
        y = lfilter(b, a, y)
    return y


def _pitch_contour(n, sr, base, wander, rng):
    t = np.arange(n) / sr
    drift = wander * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 6.28))
    drift += wander * 0.4 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    return base + drift


def _pulse_train(f0, sr, jitter, rng):
    """Impulse train at instantaneous f0 with optional jitter."""
    n = len(f0)
    out = np.zeros(n)
    phase = 0.0
    for i in range(n):
        phase += f0[i] / sr
        if jitter:
            phase += rng.normal(0, jitter) / sr
        if phase >= 1.0:
            phase -= 1.0
            out[i] = 1.0
    return out


def make_real(rng) -> np.ndarray:
    n = int(SR * DUR)
    base = rng.uniform(110, 220)
    f0 = _pitch_contour(n, SR, base, wander=12.0, rng=rng)
    src = _pulse_train(f0, SR, jitter=0.6, rng=rng)              # natural jitter
    src += 0.02 * rng.standard_normal(n)                          # breath noise
    formants = [rng.uniform(500, 800), rng.uniform(1200, 1800),
                rng.uniform(2400, 3200), rng.uniform(3400, 3900)]
    y = _formant_filter(src, formants, SR, bw=90)
    y *= (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3 * np.arange(n) / SR)))  # syllabic env
    return y


def make_fake(rng) -> np.ndarray:
    n = int(SR * DUR)
    base = rng.uniform(110, 220)
    f0 = _pitch_contour(n, SR, base, wander=3.0, rng=rng)         # flatter prosody
    src = _pulse_train(f0, SR, jitter=0.05, rng=rng)             # over-regular
    formants = [round(rng.uniform(500, 800), -1),                 # quantised formants
                round(rng.uniform(1200, 1800), -1),
                round(rng.uniform(2400, 3200), -1)]
    y = _formant_filter(src, formants, SR, bw=60)               # narrower -> smoother
    y += 0.008 * np.sin(2 * np.pi * 7000 * np.arange(n) / SR)    # vocoder buzz
    y *= (0.8 + 0.2 * np.abs(np.sin(2 * np.pi * 3 * np.arange(n) / SR)))
    return y


def _norm(y):
    peak = np.max(np.abs(y))
    return (y / peak * 0.9) if peak > 0 else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="clips per class")
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    config.REAL_DIR.mkdir(parents=True, exist_ok=True)
    config.FAKE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        sf.write(config.REAL_DIR / f"real_{i:03d}.wav", _norm(make_real(rng)), SR)
        sf.write(config.FAKE_DIR / f"fake_{i:03d}.wav", _norm(make_fake(rng)), SR)

    print(f"Wrote {args.n} real -> {config.REAL_DIR}")
    print(f"Wrote {args.n} fake -> {config.FAKE_DIR}")
    print("NOTE: synthetic demo data only. Use a real corpus for meaningful results.")


if __name__ == "__main__":
    main()
