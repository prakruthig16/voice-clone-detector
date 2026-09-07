# Voice-Clone / Impersonation Detector

Detect AI-generated (cloned / text-to-speech / voice-converted) speech versus
genuine human speech. The system extracts acoustic features that expose
synthesis artefacts and trains a machine-learning classifier to flag fakes,
reporting the metrics used in the anti-spoofing literature (F1, ROC-AUC, and
Equal Error Rate).

---

## 1. Why this works

Voice-cloning systems (Tacotron/VITS-style TTS, RVC/so-vits voice conversion,
neural vocoders like HiFi-GAN) reconstruct a waveform from a compressed
acoustic representation. That reconstruction leaves fingerprints:

- **Over-smoothed spectra** — vocoders regularise fine spectral detail, so the
  higher cepstral coefficients look "too clean".
- **Flatter prosody** — synthetic pitch contours vary less than human ones.
- **Reduced micro-variation** — natural jitter/shimmer in the glottal source is
  suppressed.
- **High-frequency buzz / phase artefacts** — characteristic of vocoder output.

We turn each clip into a fixed-length feature vector capturing these cues, then
let a classifier learn the boundary. This mirrors how ASVspoof baseline systems
work (hand-crafted features + a back-end classifier).

---

## 2. Project layout

```
voice-clone-detector/
├── config.py                 # paths, audio params, feature settings
├── requirements.txt
├── data/
│   ├── real/                 # genuine human speech (.wav/.flac/.mp3)
│   └── fake/                 # cloned / synthetic speech
├── src/
│   ├── features.py           # MFCC, spectral, prosodic, artefact features
│   ├── dataset.py            # build (X, y) matrix + feature caching
│   ├── models.py             # sklearn pipelines (gb / rf / svm)
│   ├── metrics.py            # F1, ROC-AUC, EER, confusion matrix
│   └── cnn.py                # optional PyTorch CNN on log-mel spectrograms
├── scripts/
│   ├── generate_demo_data.py # synthetic audio so it runs with no dataset
│   ├── train.py              # train + evaluate + save model & plots
│   └── detect.py             # run the trained detector on new clips
├── tests/
│   └── test_pipeline.py      # smoke tests (pytest)
└── docs/                     # generated plots, report
```

---

## 3. Setup

Requires **Python 3.9+**. From the project folder:

```bash
# (recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

`librosa` pulls in `numpy`, `scipy`, and `soundfile`. On Linux you may also need
the system library `libsndfile` (`sudo apt install libsndfile1`). On Windows and
macOS the `soundfile` wheel bundles it.

> Note: this project must be installed and run on your own machine — the build
> environment used to author it had no internet access to PyPI. All source
> files are syntax-checked and the core numeric routines (EER, feature
> summarisation) are unit-verified.

---

## 4. Quick start (no dataset needed)

The fastest way to see the full pipeline work end-to-end:

```bash
# 1. Generate synthetic demo audio (40 real + 40 fake clips)
python scripts/generate_demo_data.py --n 40

# 2. Train and evaluate
python scripts/train.py --model gb

# 3. Detect on a clip
python scripts/detect.py data/fake/fake_000.wav
python scripts/detect.py data/real/ --json
```

You should see an evaluation report (accuracy, F1, ROC-AUC, EER, confusion
matrix) and plots saved to `docs/`.

> The synthetic generator is a wiring/demo aid only. It fabricates
> "human-like" vs "vocoder-like" signals so the code runs without a corpus.
> **Do not report these numbers as real detection performance** — use a proper
> dataset (next section).

---

## 5. Using a real dataset

Replace the demo clips with real audio:

- Put **genuine human recordings** in `data/real/`
- Put **cloned / TTS / converted** audio in `data/fake/`

Any of `.wav .flac .mp3 .ogg .m4a` work; subfolders are searched recursively.

Recommended public corpora:

- **ASVspoof 2019 / 2021** — the standard spoofed-speech benchmark (bonafide vs
  spoof). Map `bonafide → data/real`, `spoof → data/fake`.
- **WaveFake** — generated speech from several neural vocoders.
- **FoR (Fake-or-Real)** — real vs synthetic speech, ready-split.
- **In-the-Wild** — real-world deepfake speech of public figures.

Aim for at least a few hundred clips per class and keep speakers balanced across
classes so the model learns *synthesis artefacts*, not *speaker identity*.

Then retrain:

```bash
python scripts/train.py --model gb --no-cache --cv 5
```

---

## 6. Command reference

**Training** (`scripts/train.py`)

| Flag | Meaning |
|------|---------|
| `--model {gb,rf,svm}` | classifier (default `gb` = GradientBoosting) |
| `--no-cache` | force feature re-extraction (use after changing data) |
| `--cv K` | also run K-fold cross-validation |

Outputs: `models/detector.joblib`, `docs/confusion_matrix.png`,
`docs/roc_curve.png`.

**Detection** (`scripts/detect.py`)

```bash
python scripts/detect.py clip.wav                 # single file
python scripts/detect.py folder/                  # every clip in a folder
python scripts/detect.py a.wav b.wav --json       # machine-readable output
python scripts/detect.py clip.wav --threshold 0.5 # override decision threshold
```

By default the decision threshold is the **EER threshold** chosen on the test
set (balances false-accepts and false-rejects). Lower it to catch more fakes at
the cost of more false alarms.

**Tests**

```bash
pytest -q
```

---

## 7. How the pieces fit

```
audio ─▶ features.extract_features ─▶ [fixed-length vector]
                                          │
data/real, data/fake ─▶ dataset.build_dataset ─▶ (X, y)
                                          │
                            models.build_model (StandardScaler + classifier)
                                          │
                    train.py ─▶ fit ─▶ metrics.evaluate ─▶ detector.joblib
                                          │
                          detect.py ─▶ load ─▶ P(fake) ─▶ verdict
```

The optional CNN (`src/cnn.py`, needs `pip install torch`) classifies log-mel
spectrograms directly and can be swapped in as an alternative back-end.

---

## 8. Extending it

- **Add features**: augment `features.extract_features` (e.g. CQCC, LFCC,
  modulation spectrum) — the rest of the pipeline adapts automatically.
- **Add a model**: extend `models.build_model` (e.g. XGBoost, LightGBM).
- **Robustness**: augment training with codec/compression and noise so the
  detector survives phone-call and social-media audio.
- **Explainability**: use permutation importance on the fitted pipeline to see
  which acoustic cues drive decisions.

---

## 9. Limitations & responsible use

- A detector trained on known vocoders may not generalise to unseen ones — this
  is an open research problem. Always validate on held-out synthesis methods.
- Compression, background noise, and short clips degrade performance.
- Treat outputs as **decision support**, not proof. Do not use this to make
  high-stakes accusations without corroborating evidence.
```
