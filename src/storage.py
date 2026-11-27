import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def count_dataset():
    """Count WAV recordings in Supabase without downloading audio."""
    bucket = get_supabase().storage.from_(BUCKET_NAME)
    counts = {}

    for folder in ("real", "fake"):
        count = 0
        offset = 0
        limit = 1000

        while True:
            files = bucket.list(folder, {"limit": limit, "offset": offset})

            count += sum(
                1 for item in files
                if item.get("name", "").lower().endswith(".wav")
            )

            if len(files) < limit:
                break

            offset += limit

        counts[folder] = count

    return counts["real"], counts["fake"]


def download_dataset(real_dir, fake_dir):
    """Download all saved training recordings from Supabase with retries."""
    client = get_supabase()
    bucket = client.storage.from_(BUCKET_NAME)

    tasks = []

    for folder, destination in (
        ("real", Path(real_dir)),
        ("fake", Path(fake_dir)),
    ):
        destination.mkdir(parents=True, exist_ok=True)

        offset = 0
        limit = 1000

        while True:
            files = bucket.list(
                folder,
                {"limit": limit, "offset": offset}
            )

            if not files:
                break

            for item in files:
                filename = item.get("name")

                if filename and filename.lower().endswith(".wav"):
                    tasks.append((
                        f"{folder}/{filename}",
                        destination / filename
                    ))

            if len(files) < limit:
                break

            offset += limit

    def download_one(task):
        storage_path, destination = task

        for attempt in range(3):
            try:
                audio_bytes = bucket.download(storage_path)
                destination.write_bytes(audio_bytes)
                return True
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    total = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_one, task)
            for task in tasks
        ]

        for future in as_completed(futures):
            future.result()
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