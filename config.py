"""Central configuration for the voice-clone / impersonation detector."""
from pathlib import Path

# ---- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
# Use the small voice dataset stored beside the Git repository
DATA_DIR = ROOT.parent / "small_voice_dataset"

REAL_DIR = DATA_DIR / "real"
FAKE_DIR = DATA_DIR / "fake"        # cloned / synthetic / spoofed speech
MODEL_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

MODEL_PATH = MODEL_DIR / "detector.joblib"
FEATURE_CACHE = MODEL_DIR / "features.npz"

# ---- Audio -------------------------------------------------------------
SAMPLE_RATE = 16000        # Hz — standard for speech / anti-spoofing
DURATION = 3.0             # seconds analysed per clip (pad/trim to this)
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# ---- Feature extraction ------------------------------------------------
N_MFCC = 20                # number of MFCC coefficients
N_FFT = 512
HOP_LENGTH = 160           # 10 ms hop at 16 kHz
WIN_LENGTH = 400           # 25 ms window

# ---- Labels ------------------------------------------------------------
LABEL_REAL = 0
LABEL_FAKE = 1
CLASS_NAMES = ["real", "fake"]

# ---- Training ----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25
