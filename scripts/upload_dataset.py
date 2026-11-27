
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import storage

DATASET_DIR = ROOT.parent / "small_voice_dataset"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def convert_to_wav(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [
        FFMPEG,
        "-y",
        "-i", str(source),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(destination),
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def upload_folder(source_folder, destination_folder, label):
    files = list(source_folder.glob("*.flac"))

    if not files:
        print(f"No FLAC files found in {source_folder}")
        return 0

    print(f"\nUploading {label}: {len(files)} recordings")

    successful = 0
    failed = 0

    for index, source in enumerate(files, start=1):
        try:
            filename = source.stem + ".wav"
            destination = destination_folder / filename

            convert_to_wav(source, destination)

            storage.upload_audio(
                destination.read_bytes(),
                label,
                filename,
            )

            successful += 1

            if index % 25 == 0:
                print(f"{label}: {index}/{len(files)} processed")

        except Exception as error:
            failed += 1
            print(f"Failed: {source.name}: {error}")

    print(f"{label} completed: {successful} uploaded, {failed} failed")
    return successful


def main():
    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR}")
        print("Place small_voice_dataset inside your project folder.")
        return

    real_count = upload_folder(
        DATASET_DIR / "real",
        config.REAL_DIR,
        "REAL",
    )

    fake_count = upload_folder(
        DATASET_DIR / "fake",
        config.FAKE_DIR,
        "FAKE",
    )

    print("\n========== UPLOAD SUMMARY ==========")
    print(f"REAL uploaded: {real_count}")
    print(f"FAKE uploaded: {fake_count}")
    print(f"TOTAL uploaded: {real_count + fake_count}")


if __name__ == "__main__":
    main()
