% Voice-Clone Detector — Step-by-Step Setup & Run Guide
% For PyCharm on Windows (beginner-friendly)
% 2026

# Read this first

This guide takes you from a fresh Windows PC to a working AI voice-clone
detector, using **PyCharm**. Follow the steps in order. Each step tells you
**what to download**, **where to get it**, and **where to put things**. You do
not need any prior Python experience.

You will do three kinds of things:

1. **Install two programs** once — Python and PyCharm.
2. **Open the project** and let PyCharm download the project's libraries.
3. **Run three commands** to generate data, train the model, and detect fakes.

Total download size is roughly **0.8–1 GB** (mostly PyCharm and the Python
libraries). A normal broadband connection handles this in a few minutes.

---

# What you will download (at a glance)

| # | Item | Where from | Size | When |
|---|------|-----------|------|------|
| 1 | Python 3.12 (Windows installer) | python.org/downloads | ~28 MB | Step 1 |
| 2 | PyCharm (free Community edition) | jetbrains.com/pycharm/download | ~400 MB | Step 2 |
| 3 | Project libraries (numpy, librosa, scikit-learn, …) | Installed automatically by pip | ~350–500 MB | Step 6 |
| 4 | *(Optional)* PyTorch — only for the deep-learning add-on | pip | ~2 GB | Step 12 |
| 5 | *(Optional)* A real speech dataset | ASVspoof / WaveFake / Fake-or-Real | varies | Step 10 |

You already have item 0: the **voice-clone-detector** project folder I created
for you.

---

# Step 1 — Install Python

Python is the language the project is written in.

1. Open your web browser and go to **https://www.python.org/downloads/**.
2. Click the big yellow button **“Download Python 3.12.x”** (any 3.11 or 3.12
   version is perfect; avoid the newest 3.13 for now to keep the audio
   libraries happy).
3. Run the downloaded file (it will be in your **Downloads** folder, named like
   `python-3.12.x-amd64.exe`).
4. **VERY IMPORTANT:** on the first install screen, tick the box at the bottom
   that says **“Add python.exe to PATH”**. Then click **“Install Now”**.
5. When it finishes, click **Close**.

**Check it worked:** press the Windows key, type `cmd`, open **Command
Prompt**, type the line below and press Enter:

```
python --version
```

You should see something like `Python 3.12.x`. If instead you see “python is not
recognized”, re-run the installer and make sure the **Add to PATH** box is
ticked (see Troubleshooting at the end).

---

# Step 2 — Install PyCharm

PyCharm is the app you'll write and run the project in. It looks and feels like
IntelliJ IDEA, so it should feel familiar.

1. Go to **https://www.jetbrains.com/pycharm/download/**.
2. Scroll down to **“PyCharm Community Edition”** — this one is **free**.
   Click **Download** (choose the **Windows** version).

   > If the page only shows one big “Download” button, that's fine — the free
   > tier is included. You do **not** need a paid licence for this project.
3. Run the downloaded installer (`pycharm-community-…​.exe`) and click through
   with the default options. It's fine to tick **“Create Desktop Shortcut”**.
4. Launch PyCharm when it's done. Accept the privacy notice; you can skip
   signing in.

---

# Step 3 — Put the project folder somewhere sensible

You should already have the **voice-clone-detector** folder (the one containing
`config.py`, `README.md`, and the `src`, `scripts`, `docs` sub-folders).

1. Create a folder for your projects if you don't have one, e.g.
   `C:\Users\<you>\PycharmProjects\`.
2. **Move or copy the whole `voice-clone-detector` folder into it.** Keep the
   inside structure exactly as it is — don't rename or move the sub-folders,
   because the code expects `src/`, `scripts/`, and `data/` to sit where they
   are.

Final location will look like:

```
C:\Users\<you>\PycharmProjects\voice-clone-detector\
    config.py
    requirements.txt
    README.md
    src\
    scripts\
    data\
    docs\
    tests\
```

---

# Step 4 — Open the project in PyCharm

1. Open PyCharm. On the welcome screen click **“Open”** (not “New Project”).
2. Browse to your `voice-clone-detector` folder, select it, and click **OK**.
3. If PyCharm asks **“Trust project?”**, click **Trust Project**.

You'll now see the project files on the left.

---

# Step 5 — Let PyCharm create a “virtual environment”

A *virtual environment* (venv) is a private box that holds this project's
libraries so they don't clash with anything else on your PC. PyCharm sets it up
for you.

1. When you open the project, PyCharm usually shows a small notification:
   **“Creating virtual environment”** or offers to configure a Python
   interpreter — let it. If it asks which base Python to use, pick the
   **Python 3.12** you installed in Step 1, then click **OK**.
2. If you *don't* see that prompt, set it up manually:
   - Open **File ▸ Settings ▸ Project: voice-clone-detector ▸ Python
     Interpreter**.
   - Click **“Add Interpreter” ▸ “Add Local Interpreter”**.
   - Choose **Virtualenv ▸ New**, make sure the base interpreter is your
     Python 3.12, and click **OK**.

You'll know it worked when the bottom-right of the PyCharm window shows
something like **“Python 3.12 (voice-clone-detector)”**.

---

# Step 6 — Install the project's libraries

This downloads the tools the detector needs: numpy, scipy, **librosa** (audio),
**scikit-learn** (machine learning), soundfile, joblib, matplotlib, pandas, and
tqdm. They're all listed in `requirements.txt`.

**Easiest way (PyCharm banner):** when you open `requirements.txt`, PyCharm
shows a yellow bar at the top saying **“Install requirements”**. Click it and
wait.

**Or use the built-in Terminal:**

1. Open the terminal inside PyCharm: **View ▸ Tool Windows ▸ Terminal** (it
   opens already inside your project folder).
2. Type this and press Enter:

```
pip install -r requirements.txt
```

3. Wait for it to finish. **librosa** pulls in a few large pieces (numba,
   llvmlite), so this step can take **2–5 minutes** and print a lot of text.
   That's normal. It's done when you get your prompt back with no red
   **ERROR** lines.

**Check it worked** — in the same terminal:

```
python -c "import librosa, sklearn, soundfile; print('all good')"
```

If it prints `all good`, you're ready.

> Windows note: the audio library `soundfile` includes everything it needs, so
> you do **not** have to install anything extra for `.wav` files to work.

---

# Step 7 — Generate demo audio (so you can test right away)

You don't need to find a dataset yet — the project can make its own practice
audio. In the PyCharm terminal, run:

```
python scripts/generate_demo_data.py --n 40
```

**What this does:** creates 40 “real-sounding” clips and 40 “fake-sounding”
clips.
**Where they go:** into `data/real\` and `data/fake\` inside your project.
You'll see them appear in the file list on the left.

> These synthetic clips are only for testing that everything runs. They are
> **not** real speech — see Step 10 for using a proper dataset.

---

# Step 8 — Train the detector

Now train the machine-learning model on those clips:

```
python scripts/train.py --model gb
```

**What you'll see:** a short report in the terminal with **Accuracy, Precision,
Recall, F1, ROC-AUC, and EER** (Equal Error Rate — lower is better).

**What it creates:**

- `models\detector.joblib` — the trained model (this is the “brain”).
- `docs\confusion_matrix.png` and `docs\roc_curve.png` — result charts you can
  open by double-clicking them in PyCharm.

---

# Step 9 — Detect whether a clip is fake

Point the trained detector at an audio file:

```
python scripts/detect.py data/fake/fake_000.wav
```

You'll get a line like:

```
fake_000.wav      P(fake)=0.93  ->  FAKE (cloned/synthetic)
```

Try a real one too:

```
python scripts/detect.py data/real/real_000.wav
```

You can also scan a whole folder at once:

```
python scripts/detect.py data/fake/
```

---

# Step 10 — Use REAL data (for meaningful results)

The demo clips prove the pipeline works, but for a real project you want genuine
recordings vs genuine AI-cloned speech.

**Where to put the audio:**

- Put real human recordings into `data\real\`
- Put cloned / text-to-speech / voice-converted audio into `data\fake\`

`.wav`, `.flac`, and `.mp3` all work.

**Where to get datasets** (these are the standard research ones — most require a
quick free sign-up):

- **ASVspoof 2019 / 2021** — the standard benchmark. Its `bonafide` files go in
  `data\real\`, its `spoof` files go in `data\fake\`.
- **WaveFake** — speech made by several AI vocoders → `data\fake\`.
- **Fake-or-Real (FoR)** — comes already split into real and fake.

After adding real audio, delete the demo clips (or start with empty folders) and
retrain, forcing a fresh read of the new files:

```
python scripts/train.py --model gb --no-cache
```

---

# Step 11 — (Optional) Run the built-in tests

To confirm the code is healthy:

```
pip install pytest
pytest -q
```

All tests should pass.

---

# Step 12 — (Optional) The deep-learning add-on

The project includes a neural-network detector as an alternative. It needs
**PyTorch**, which is a large (~2 GB) download, so only do this if you want it:

```
pip install torch
```

You can skip this entirely — the main detector in Steps 7–9 does not need it.

---

# Running commands with the green ▶ button (optional)

You've been using the terminal, which is the simplest way. If you prefer
PyCharm's **Run** button:

1. Open `scripts\train.py`.
2. Top-right, click the dropdown next to the run icons ▸ **“Edit
   Configurations…”**.
3. In **Parameters**, type the arguments (e.g. `--model gb`).
4. Set **Working directory** to your project's top folder
   (`…\voice-clone-detector`).
5. Click **OK**, then the green ▶ to run.

The terminal method from the earlier steps is usually faster for one-off runs.

---

# Troubleshooting

**“python is not recognized”** — Python isn't on your PATH. Re-run the Python
installer from Step 1, choose **Modify**, and make sure **“Add python.exe to
PATH”** is ticked. Then close and reopen PyCharm.

**`pip install` fails or is very slow** — first upgrade pip, then try again:
`python -m pip install --upgrade pip`. A slow, quiet spell while installing
librosa/numba is normal, not a failure.

**Red “ERROR” about llvmlite / numba while installing** — this usually means
your Python is too new. Use Python **3.11 or 3.12** (not 3.13), then recreate
the virtual environment (Step 5) and reinstall (Step 6).

**“ModuleNotFoundError: No module named 'src'”** — you're running from the wrong
folder. Make sure the PyCharm terminal prompt shows you're inside
`…\voice-clone-detector` before running `python scripts/...`.

**“No trained model … Run scripts/train.py first”** when detecting — you skipped
Step 8. Train first, then detect.

**A clip won't load / “error” in detect output** — the file may be corrupt or an
unusual format. Convert it to `.wav` and try again.

---

# Quick recap (the whole thing in 5 lines)

```
1. Install Python 3.12  (tick "Add to PATH")
2. Install PyCharm Community, then Open the project folder
3. Let PyCharm make a venv, then:  pip install -r requirements.txt
4. python scripts/generate_demo_data.py --n 40
5. python scripts/train.py --model gb   →   python scripts/detect.py data/fake/fake_000.wav
```

That's it — you now have a working AI voice-clone detector.
