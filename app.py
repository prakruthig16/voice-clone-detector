
from pathlib import Path
import subprocess
import tempfile
import time
import sys

import joblib
import streamlit as st
import imageio_ffmpeg
from streamlit_mic_recorder import mic_recorder

import config
from src.features import extract_features
from src import storage


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Voice Clone Detector",
    page_icon="🎙️",
    layout="wide",
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(34, 211, 238, 0.13), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(139, 92, 246, 0.16), transparent 30%),
        linear-gradient(135deg, #080d1c 0%, #101a32 52%, #111827 100%);
    color: #edf4ff;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 28, 0.72);
}

[data-testid="stSidebar"] {
    background: #0b1224;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 42px 28px;
    border: 1px solid rgba(125, 211, 252, 0.28);
    border-radius: 26px;
    background: linear-gradient(120deg, #172554 0%, #312e81 52%, #6d28d9 100%);
    text-align: center;
    margin: 4px 0 30px 0;
    box-shadow: 0 18px 55px rgba(2, 6, 23, 0.38);
}

.hero:before {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -70px;
    top: -145px;
    border-radius: 50%;
    background: rgba(103, 232, 249, 0.18);
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(2rem, 4vw, 3.1rem) !important;
    font-weight: 800 !important;
    letter-spacing: -1.2px;
    margin-bottom: 8px;
}

.hero p {
    color: #dbeafe !important;
    font-size: 1.08rem;
    margin: 0;
}

h1, h2, h3, h4 {
    color: #f4f7ff !important;
    font-weight: 750 !important;
}

p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: #d5deef;
}

[data-testid="stCaptionContainer"] {
    color: #9fb0ca !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 9px;
    padding: 8px;
    border: 1px solid rgba(148, 163, 184, 0.17);
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.78);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 18px;
    color: #cbd5e1;
    border-radius: 11px;
    font-weight: 650;
    background: transparent;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(100deg, #2563eb, #7c3aed) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: transparent;
}

div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stMetric"],
div[data-testid="stExpander"] {
    border: 1px solid rgba(148, 163, 184, 0.20);
    border-radius: 17px;
    background: rgba(15, 23, 42, 0.72);
}

div[data-testid="stMetric"] {
    padding: 17px 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetricLabel"] {
    color: #a9bbd6 !important;
}

[data-testid="stMetricValue"] {
    color: #67e8f9 !important;
    font-weight: 800;
}

.stButton > button, .stDownloadButton > button {
    border: 0;
    border-radius: 12px;
    min-height: 44px;
    padding: 0.55rem 1.15rem;
    color: #ffffff;
    font-weight: 700;
    background: linear-gradient(105deg, #2563eb, #7c3aed);
    box-shadow: 0 7px 18px rgba(37, 99, 235, 0.22);
    transition: transform 0.15s ease, filter 0.15s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover {
    color: #ffffff;
    filter: brightness(1.13);
    transform: translateY(-2px);
    border: 0;
}

.stButton > button:focus, .stDownloadButton > button:focus {
    color: #ffffff;
    border: 1px solid #67e8f9;
}

[data-testid="stFileUploader"] section {
    border: 1px dashed rgba(103, 232, 249, 0.55);
    border-radius: 15px;
    background: rgba(15, 23, 42, 0.62);
}

[data-testid="stFileUploader"] button {
    border-radius: 9px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background: #111c33 !important;
    color: #f8fafc !important;
    border-radius: 10px !important;
}

.stRadio [role="radiogroup"] {
    gap: 12px;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #22d3ee, #8b5cf6);
}

hr {
    border-color: rgba(148, 163, 184, 0.20);
}

[data-testid="stAlert"] {
    border-radius: 13px;
    border: 1px solid rgba(148, 163, 184, 0.22);
}

audio {
    width: 100%;
    border-radius: 12px;
}

.footer-note {
    text-align: center;
    color: #91a4c1;
    padding: 14px 0 4px 0;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
<h1>🎙️ AI Voice Clone Detector</h1>
<p>Detect synthetic voices • Record live audio • Train your own voice model</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "wma"}
VIDEO_EXTENSIONS = {"mp4", "mov", "mkv", "avi", "webm"}

SUPPORTED_EXTENSIONS = sorted(
    AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
)

MAX_UPLOAD_MB = 100


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------
@st.cache_resource
def load_detector():
    """
    Always try to retrieve the latest model from Supabase.
    If Supabase has no model yet, use an existing local model.
    """

    downloaded = storage.download_model(config.MODEL_PATH)

    if not config.MODEL_PATH.exists():
        if not downloaded:
            raise FileNotFoundError(
                "No trained model found. Please train the model first."
            )

    bundle = joblib.load(config.MODEL_PATH)

    return (
        bundle["pipeline"],
        bundle.get("eer_threshold", 0.5)
    )


def refresh_model():
    load_detector.clear()


# --------------------------------------------------
# AUDIO CONVERSION
# --------------------------------------------------
def convert_to_wav(uploaded_file):
    """Convert uploaded audio/video into mono 16 kHz WAV."""

    suffix = Path(uploaded_file.name).suffix.lower() or ".audio"

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = Path(temp_dir) / ("input" + suffix)
        output_path = Path(temp_dir) / "converted.wav"

        input_path.write_bytes(uploaded_file.getvalue())

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", str(config.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0 or not output_path.exists():
            raise ValueError(
                "Could not convert this file. Check whether it contains "
                "audio and is not corrupted."
            )

        return output_path.read_bytes()


def convert_recording_to_wav(audio_bytes):
    """Convert microphone WebM recording into mono 16 kHz WAV."""

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = Path(temp_dir) / "recording_input.webm"
        output_path = Path(temp_dir) / "recording_converted.wav"

        input_path.write_bytes(audio_bytes)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        command = [
            ffmpeg,
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", str(config.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(output_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0 or not output_path.exists():
            raise ValueError(
                "Could not process the microphone recording."
            )

        return output_path.read_bytes()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_wav_bytes(wav_bytes):
    """Extract audio features and calculate fake voice probability."""

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp:

        temp.write(wav_bytes)
        temp_path = Path(temp.name)

    try:
        model, threshold = load_detector()

        features = extract_features(
            str(temp_path)
        ).reshape(1, -1)

        p_fake = float(
            model.predict_proba(features)[0, config.LABEL_FAKE]
        )

        return p_fake, threshold

    finally:
        temp_path.unlink(missing_ok=True)


def show_prediction(wav_bytes):

    p_fake, threshold = predict_wav_bytes(wav_bytes)

    st.audio(wav_bytes, format="audio/wav")

    st.subheader("Detection Result")

    if p_fake >= threshold:
        st.error("⚠️ AI-CLONED / SYNTHETIC")
    else:
        st.success("✅ GENUINE / REAL")

    st.metric(
        "Estimated AI-generated probability",
        f"{p_fake * 100:.2f}%"
    )

    st.progress(
        min(max(p_fake, 0.0), 1.0)
    )

    st.caption(
        f"Decision threshold: {threshold:.4f}"
    )

    st.warning(
        "This is a model estimate, not proof of authenticity. "
        "Performance depends on the quality and diversity of training data."
    )


# --------------------------------------------------
# DATASET MANAGER
# --------------------------------------------------
def dataset_counts():
    """
    Download recordings from Supabase before counting.
    This keeps the dataset count consistent with cloud storage.
    """

    config.REAL_DIR.mkdir(parents=True, exist_ok=True)
    config.FAKE_DIR.mkdir(parents=True, exist_ok=True)

    storage.download_dataset(
        config.REAL_DIR,
        config.FAKE_DIR
    )

    real_count = sum(
        1
        for f in config.REAL_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() == ".wav"
    )

    fake_count = sum(
        1
        for f in config.FAKE_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() == ".wav"
    )

    return real_count, fake_count


def save_training_audio(wav_bytes, label, original_name):
    """
    Save a recording locally and upload it permanently to Supabase.
    """

    if label not in ("REAL", "FAKE"):
        raise ValueError("Invalid recording label.")

    folder = (
        config.REAL_DIR
        if label == "REAL"
        else config.FAKE_DIR
    )

    folder.mkdir(parents=True, exist_ok=True)

    safe_name = Path(original_name).stem

    safe_name = "".join(
        c for c in safe_name
        if c.isalnum() or c in "-_"
    )[:40]

    filename = f"{safe_name}_{time.time_ns()}.wav"

    # Save locally for the current training session.
    destination = folder / filename
    destination.write_bytes(wav_bytes)

    # Upload permanently to Supabase.
    storage.upload_audio(
        wav_bytes,
        label,
        filename
    )

    # Remove old feature cache.
    if config.FEATURE_CACHE.exists():
        config.FEATURE_CACHE.unlink()

    return destination


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
def train_model():

    # Download all permanently stored recordings.
    storage.download_dataset(
        config.REAL_DIR,
        config.FAKE_DIR
    )

    script = (
        Path(__file__).parent
        / "scripts"
        / "train.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--no-cache"
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )

    if result.returncode == 0:

        # Upload the newly trained model to Supabase.
        storage.upload_model(config.MODEL_PATH)

    # Clear the cached old model.
    refresh_model()

    return result


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
tab_detect, tab_add, tab_dataset, tab_train = st.tabs([
    "🔍 Voice Detection",
    "➕ Add Training Voices",
    "📁 Dataset Manager",
    "🧠 Train Model",
])


# ==================================================
# TAB 1: VOICE DETECTION
# ==================================================
with tab_detect:

    st.header("Voice Detection")

    st.subheader("📁 Detect from an uploaded file")

    st.write(
        "Upload an audio or video file for analysis."
    )

    detection_file = st.file_uploader(
        "Choose audio or video",
        type=SUPPORTED_EXTENSIONS,
        key="detection_upload",
    )

    if detection_file:

        if detection_file.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error("Maximum file size is 100 MB.")

        else:

            st.success("File uploaded successfully.")

            st.write("Filename:", detection_file.name)

            st.write(
                f"Size: {detection_file.size / (1024 * 1024):.2f} MB"
            )

            if st.button(
                "Analyze Uploaded Voice",
                type="primary",
                key="analyze_upload"
            ):

                try:

                    with st.spinner(
                        "Extracting audio and analyzing..."
                    ):

                        wav_bytes = convert_to_wav(
                            detection_file
                        )

                        show_prediction(wav_bytes)

                except Exception as error:

                    st.error(
                        f"Detection failed: {error}"
                    )

    st.markdown("---")

    st.subheader("🎙️ Live Microphone Recording")

    st.write(
        "Allow microphone access in your browser, "
        "record your voice, then analyze it."
    )

    recording = mic_recorder(
        start_prompt="🔴 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=True,
        format="webm",
        key="live_recorder",
    )

    if recording and recording.get("bytes"):

        st.session_state["live_recording_bytes"] = (
            recording["bytes"]
        )

    recorded_bytes = st.session_state.get(
        "live_recording_bytes"
    )

    if recorded_bytes:

        st.success("Recording received!")

        st.audio(
            recorded_bytes,
            format="audio/webm"
        )

        if st.button(
            "Analyze Live Recording",
            type="primary",
            key="analyze_live"
        ):

            try:

                with st.spinner(
                    "Converting recording and analyzing..."
                ):

                    normalized_bytes = convert_recording_to_wav(
                        recorded_bytes
                    )

                    show_prediction(normalized_bytes)

            except Exception as error:

                st.error(
                    f"Live detection failed: {error}"
                )

        st.download_button(
            "Download Live Recording",
            data=recorded_bytes,
            file_name="live_recording.webm",
            mime="audio/webm",
        )

        if st.button(
            "Clear Recording",
            key="clear_live_recording"
        ):

            st.session_state.pop(
                "live_recording_bytes",
                None
            )

            st.rerun()


# ==================================================
# TAB 2: ADD TRAINING DATA
# ==================================================
with tab_add:

    st.header("Add Training Voices")

    st.write(
        "Upload recordings to expand the training dataset. "
        "Choose the correct label based on verified source information."
    )

    training_files = st.file_uploader(
        "Upload multiple audio or video files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="training_upload",
    )

    training_label = st.radio(
        "Choose the verified label for these recordings",
        ["REAL", "FAKE"],
        horizontal=True,
        key="training_label",
    )

    if training_files:

        st.write(
            f"Selected {len(training_files)} file(s)."
        )

        if st.button(
            "Convert and Save to Dataset",
            type="primary"
        ):

            saved = 0

            progress = st.progress(0)

            for index, uploaded in enumerate(training_files):

                try:

                    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise ValueError(
                            "File exceeds 100 MB."
                        )

                    wav_bytes = convert_to_wav(
                        uploaded
                    )

                    save_training_audio(
                        wav_bytes,
                        training_label,
                        uploaded.name
                    )

                    saved += 1

                except Exception as error:

                    st.error(
                        f"{uploaded.name}: {error}"
                    )

                progress.progress(
                    (index + 1) / len(training_files)
                )

            if saved:

                st.success(
                    f"Successfully saved {saved} recording(s) as {training_label}."
                )

                st.info(
                    "Your recordings have been uploaded to Supabase. "
                    "Retrain the model to use them."
                )

                st.rerun()


# ==================================================
# TAB 3: DATASET MANAGER
# ==================================================
with tab_dataset:

    st.header("Dataset Manager")

    try:

        real_count, fake_count = dataset_counts()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "REAL recordings",
            real_count
        )

        col2.metric(
            "FAKE recordings",
            fake_count
        )

        col3.metric(
            "Total recordings",
            real_count + fake_count
        )

        st.markdown("---")

        st.subheader("Dataset distribution")

        if real_count + fake_count:

            st.bar_chart({
                "Recordings": {
                    "REAL": real_count,
                    "FAKE": fake_count
                }
            })

        else:

            st.info("No recordings found.")

        st.caption(
            "Only WAV files are counted because training uploads "
            "are converted to WAV."
        )

    except Exception as error:

        st.error(
            f"Could not load dataset from Supabase: {error}"
        )

    if st.button("Refresh Dataset"):

        st.rerun()


# ==================================================
# TAB 4: TRAIN MODEL
# ==================================================
with tab_train:

    st.header("Train / Retrain Model")

    try:

        real_count, fake_count = dataset_counts()

        st.write(
            "The model will learn from all labeled recordings "
            "currently stored in your dataset."
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "REAL samples",
            real_count
        )

        col2.metric(
            "FAKE samples",
            fake_count
        )

        if real_count < 2 or fake_count < 2:

            st.warning(
                "Add at least two recordings in each class before training."
            )

        else:

            st.warning(
                "Training replaces the currently saved model. "
                "The new model will also be uploaded to Supabase."
            )

            if st.button(
                "Start Model Training",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Training model. Please wait..."
                    ):

                        result = train_model()

                    if result.returncode == 0:

                        st.success(
                            "Model trained and uploaded to Supabase successfully!"
                        )

                        st.subheader("Training Report")

                        st.code(
                            result.stdout,
                            language="text"
                        )

                        plot_names = [
                            "confusion_matrix.png",
                            "roc_curve.png",
                            "precision_recall.png",
                            "precision_recall_curve.png",
                            "learning_curve.png",
                        ]

                        for plot_name in plot_names:

                            plot_path = (
                                config.DOCS_DIR / plot_name
                            )

                            if plot_path.exists():

                                st.image(
                                    str(plot_path),
                                    caption=plot_name.replace(
                                        "_", " "
                                    ).title()
                                )

                    else:

                        st.error(
                            "Training failed."
                        )

                        st.code(
                            result.stdout + "\n" + result.stderr,
                            language="text"
                        )

                except subprocess.TimeoutExpired:

                    st.error(
                        "Training took too long and was stopped."
                    )

                except Exception as error:

                    st.error(
                        f"Training error: {error}"
                    )

    except Exception as error:

        st.error(
            f"Could not load training dataset: {error}"
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div class="footer-note">
        🎙️ AI Voice Clone Detector
        &nbsp;|&nbsp;
        Detection • Dataset • Training • Cloud Storage
    </div>
    """,
    unsafe_allow_html=True
)
