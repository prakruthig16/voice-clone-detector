
import os
from pathlib import Path

import streamlit as st
from supabase import create_client


BUCKET_NAME = "voice-data"


def get_supabase():
    """Connect to Supabase using Streamlit secrets or environment variables."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Supabase credentials are missing.")

    return create_client(url, key)


def upload_audio(wav_bytes, label, filename):
    """Permanently upload a WAV recording to the appropriate folder."""
    if label not in ("REAL", "FAKE"):
        raise ValueError("Label must be REAL or FAKE.")

    folder = "real" if label == "REAL" else "fake"
    storage_path = f"{folder}/{filename}"

    client = get_supabase()
    client.storage.from_(BUCKET_NAME).upload(
        path=storage_path,
        file=wav_bytes,
        file_options={
            "content-type": "audio/wav",
            "upsert": "true",
        },
    )

    return storage_path


def download_dataset(real_dir, fake_dir):
    """Download all saved training recordings into local training folders."""
    client = get_supabase()
    bucket = client.storage.from_(BUCKET_NAME)

    total = 0

    for folder, destination in (
        ("real", Path(real_dir)),
        ("fake", Path(fake_dir)),
    ):
        destination.mkdir(parents=True, exist_ok=True)

        files = bucket.list(folder)

        for item in files:
            filename = item.get("name")

            if not filename or not filename.lower().endswith(".wav"):
                continue

            storage_path = f"{folder}/{filename}"
            audio_bytes = bucket.download(storage_path)

            (destination / filename).write_bytes(audio_bytes)
            total += 1

    return total


def upload_model(model_path):
    """Save the newly trained model permanently in Supabase."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    client = get_supabase()
    client.storage.from_(BUCKET_NAME).upload(
        path="models/detector.joblib",
        file=model_path.read_bytes(),
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true",
        },
    )


def download_model(model_path):
    """Download the latest trained model from Supabase."""
    client = get_supabase()

    try:
        model_bytes = client.storage.from_(BUCKET_NAME).download(
            "models/detector.joblib"
        )
    except Exception:
        return False

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(model_bytes)

    return True
