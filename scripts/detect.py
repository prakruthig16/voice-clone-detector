"""
Run the trained detector on one or more audio files.

Usage:
    python scripts/detect.py path/to/clip.wav
    python scripts/detect.py clip1.wav clip2.wav --json
    python scripts/detect.py folder/            # all audio in a folder
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.dataset import AUDIO_EXTS
from src.features import extract_features


def _gather(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pth = Path(p)
        if pth.is_dir():
            out += sorted(f for f in pth.rglob("*")
                          if f.suffix.lower() in AUDIO_EXTS)
        else:
            out.append(pth)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="audio files or folders")
    ap.add_argument("--json", action="store_true", help="output JSON")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override P(fake) decision threshold")
    args = ap.parse_args()

    if not config.MODEL_PATH.exists():
        sys.exit(f"No trained model at {config.MODEL_PATH}. Run scripts/train.py first.")

    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["pipeline"]
    thr = args.threshold if args.threshold is not None \
        else bundle.get("eer_threshold", 0.5)

    results = []
    for f in _gather(args.paths):
        try:
            feat = extract_features(str(f)).reshape(1, -1)
            p_fake = float(model.predict_proba(feat)[0, config.LABEL_FAKE])
            verdict = "FAKE (cloned/synthetic)" if p_fake >= thr else "REAL (genuine)"
            results.append({"file": str(f), "p_fake": round(p_fake, 4),
                            "verdict": verdict})
        except Exception as e:  # noqa: BLE001
            results.append({"file": str(f), "error": str(e)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nDecision threshold P(fake) >= {thr:.3f}\n" + "-" * 60)
        for r in results:
            if "error" in r:
                print(f"  {Path(r['file']).name:<30} ERROR: {r['error']}")
            else:
                print(f"  {Path(r['file']).name:<30} "
                      f"P(fake)={r['p_fake']:.3f}  ->  {r['verdict']}")
        print("-" * 60)


if __name__ == "__main__":
    main()
