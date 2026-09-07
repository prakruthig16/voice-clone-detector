% Detecting AI-Powered Voice Cloning and Impersonation in Speech
% A Feature-Based Machine-Learning Approach
% 2026

# Abstract

The rapid maturation of neural text-to-speech (TTS) and voice-conversion
systems has made it possible to clone a person's voice from only a few seconds
of reference audio. While this technology enables valuable applications, it has
also fuelled a sharp rise in voice-based fraud, misinformation, and identity
impersonation. This project presents a complete, reproducible system for
distinguishing genuine human speech from AI-generated (cloned or synthetic)
speech. The system extracts a compact set of acoustic features — cepstral,
spectral, prosodic, and vocoder-artefact descriptors — that expose the
fingerprints left by speech-synthesis pipelines, and trains a classical
machine-learning classifier to separate the two classes. We report the metrics
standard to the anti-spoofing literature: precision, recall, F1-score,
ROC-AUC, and the Equal Error Rate (EER). The design is modular: feature
extraction, dataset assembly, modelling, and evaluation are cleanly separated,
and an optional convolutional neural network over log-mel spectrograms is
provided as an alternative back-end. The result is an end-to-end pipeline
suitable both as an academic study of spoofed-speech detection and as a
practical starting point for deployment.

# 1. Introduction

## 1.1 Motivation

Voice is increasingly used as a biometric and as a channel of trust — in
banking authentication, in customer service, and in personal communication. At
the same time, systems such as Tacotron 2, VITS, and neural vocoders like
HiFi-GAN, together with voice-conversion frameworks such as RVC and so-vits,
can synthesise convincing speech in a target speaker's voice. This asymmetry
creates a security gap: the same audio that a human finds convincing may
contain machine-detectable artefacts.

The goal of this project is to build a detector that classifies a short speech
clip as **real** (bonafide human speech) or **fake** (cloned, text-to-speech,
or voice-converted speech), and to do so in a way that is transparent,
reproducible, and extensible.

## 1.2 Problem statement

Given an audio recording of speech, decide whether the waveform was produced by
a genuine human vocal tract or reconstructed by a speech-synthesis system. This
is framed as a binary classification problem with a probabilistic output, so
that a tunable decision threshold can trade off false accepts against false
rejects.

## 1.3 Objectives

The project sets out to: (i) design acoustic features that capture synthesis
artefacts; (ii) implement a robust, cached feature-extraction and dataset
pipeline; (iii) train and evaluate a classifier using anti-spoofing metrics;
(iv) provide an inference tool that scores arbitrary clips; and (v) document the
system so it can be rebuilt and extended.

# 2. Background and Related Work

Spoofed-speech detection has been studied intensively through the ASVspoof
challenge series (2015–2021), which established shared datasets and the Equal
Error Rate as the primary metric. Baseline systems in this literature typically
combine hand-crafted front-end features — such as Constant-Q Cepstral
Coefficients (CQCC), Linear-Frequency Cepstral Coefficients (LFCC), or
Mel-Frequency Cepstral Coefficients (MFCC) — with a back-end classifier such as
a Gaussian Mixture Model or, more recently, a neural network.

The key insight exploited across this body of work is that vocoders and neural
synthesisers do not perfectly reproduce the fine structure of natural speech.
They tend to over-smooth spectral detail, flatten prosody, and introduce
characteristic high-frequency and phase artefacts. Our feature design follows
this established intuition while keeping the model interpretable.

# 3. Methodology

## 3.1 System overview

The pipeline transforms each audio clip into a fixed-length feature vector,
assembles a labelled matrix from the `real` and `fake` folders, standardises the
features, and fits a classifier. At inference time a new clip is passed through
the identical feature extractor and scored, yielding a probability that the clip
is fake.

The processing chain is: audio → feature extraction → standardisation →
classifier → probability of fake → thresholded verdict.

## 3.2 Audio pre-processing

All audio is resampled to 16 kHz mono and fixed to a 3-second analysis window
(padded or trimmed), which matches common anti-spoofing practice and bounds the
feature dimensionality. Each clip is peak-normalised so that loudness
differences do not leak into the classifier as a spurious cue.

## 3.3 Feature extraction

Four complementary families of features are computed at the frame level and then
summarised over time by their mean and standard deviation, producing one
fixed-length vector per clip regardless of clip length:

**Cepstral features.** Twenty MFCCs together with their first and second time
derivatives (deltas). MFCCs summarise the spectral envelope; the deltas capture
its temporal dynamics. Synthesis systems tend to regularise the higher-order
coefficients, so their distribution differs from natural speech.

**Spectral-shape features.** Spectral centroid, bandwidth, roll-off, flatness,
seven-band spectral contrast, and zero-crossing rate. These describe the
distribution of energy across frequency and are sensitive to the over-smoothing
introduced by vocoders.

**Prosodic features.** Fundamental frequency (F0) statistics — mean, standard
deviation, minimum, and maximum — estimated with the probabilistic YIN
algorithm, plus the voiced ratio. Cloned speech frequently exhibits flatter,
more regular intonation, which shows up as reduced F0 variance.

**Artefact features.** Energy dynamics (RMS) and the variance of the log-mel
delta, which acts as a proxy for vocoder buzz and frame-to-frame instability.

The concatenated vector has a fixed dimensionality determined only by the
configuration, so any clip maps to the same feature space.

## 3.4 Classification models

The primary back-end is a scikit-learn pipeline consisting of a
`StandardScaler` followed by a classifier. Three interchangeable classifiers are
provided: Gradient Boosting (the default, strong on tabular acoustic features),
Random Forest (a fast, robust baseline), and an RBF-kernel Support Vector
Machine with probability estimates. Standardisation is fitted only on training
data, inside the pipeline, to avoid information leakage.

An optional deep-learning back-end — a compact convolutional neural network
operating on log-mel spectrograms — is included for experimentation and is
activated only when PyTorch is installed, so the core system has no
deep-learning dependency.

## 3.5 Evaluation protocol

Data is split into stratified training and test sets (75/25 by default), with
optional k-fold cross-validation. We report accuracy, precision, recall, and
F1-score for the fake class, ROC-AUC, and the Equal Error Rate. The EER is the
operating point at which the false-acceptance and false-rejection rates are
equal; it is the standard single-number summary in anti-spoofing because it is
threshold-independent. The EER threshold is stored with the model and used as
the default decision boundary at inference time.

# 4. Implementation

The system is implemented in Python. `librosa` provides the signal-processing
primitives, `scikit-learn` the modelling and metrics, and `matplotlib` the
diagnostic plots. The codebase is organised into a feature module, a dataset
module with feature caching (keyed by a fingerprint of the input files), a
models module, and a metrics module, orchestrated by `train.py` and
`detect.py` command-line scripts.

Because a curated corpus is not always immediately available, the project also
ships a synthetic data generator. It produces two deliberately distinguishable
but non-trivial signal classes using a source-filter model: a "human-like"
class with natural pitch wander, jitter, and drifting formants, and a
"vocoder-like" class with over-regular excitation, quantised formants, and a
faint high-frequency buzz. This lets the entire pipeline be exercised
end-to-end without downloading data, while clearly not being a substitute for a
real dataset.

The correctness of the core numerical routines was verified independently of
the third-party stack: the Equal Error Rate implementation returns
approximately zero on perfectly separable scores and approximately 0.5 on random
scores, and the feature-summarisation routine was confirmed to yield a
fixed-length, finite vector irrespective of input length.

# 5. Results and Discussion

On the synthetic demonstration data the pipeline runs end-to-end and produces a
full evaluation report and diagnostic plots (confusion matrix and ROC curve).
Because the synthetic classes are generated to be separable, the demonstration
numbers are high by construction and are not reported as scientific results —
their purpose is solely to validate that data flows correctly from audio to
verdict.

For meaningful evaluation, the system is intended to be trained on an
established corpus such as ASVspoof 2019/2021, WaveFake, or Fake-or-Real, with
the bonafide and spoof partitions mapped to the `real` and `fake` folders.
Balancing speakers across the two classes is important so that the classifier
learns synthesis artefacts rather than speaker identity. Under such a protocol,
the reported F1, ROC-AUC, and EER characterise genuine detection performance.

A central limitation, well documented in the literature, is cross-method
generalisation: a detector trained on a fixed set of vocoders may degrade on
unseen synthesis methods. Compression, background noise, and very short clips
also reduce accuracy. These observations motivate the robustness extensions
discussed below.

# 6. Conclusion and Future Work

This project delivers a complete, modular, and reproducible system for detecting
AI-powered voice cloning and impersonation, grounded in the feature families and
evaluation metrics established by the anti-spoofing community. The architecture
cleanly separates feature extraction, modelling, and evaluation, making it
straightforward to extend.

Future work includes adding complementary front-end features (CQCC, LFCC,
modulation spectrum), training the convolutional back-end and comparing it with
the classical models, introducing data augmentation with codec and noise
degradations to improve robustness to real-world audio, evaluating explicitly on
held-out synthesis methods to measure generalisation, and applying
permutation-importance analysis to interpret which acoustic cues most influence
the decision.

## Responsible use

The detector should be treated as decision support rather than proof. Given the
known limitations around unseen synthesis methods and degraded audio, its
outputs should not be used to make high-stakes accusations without corroborating
evidence.

# References

1. Todisco, M. et al. *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio
   Detection.* Interspeech, 2019.
2. Yamagishi, J. et al. *ASVspoof 2021: Accelerating Progress in Spoofed and
   Deepfake Speech Detection.* 2021.
3. Frank, J. and Schönherr, L. *WaveFake: A Data Set to Facilitate Audio
   Deepfake Detection.* NeurIPS Datasets and Benchmarks, 2021.
4. Reimao, R. and Tzerpos, V. *FoR: A Dataset for Synthetic Speech Detection.*
   SpeD, 2019.
5. Kong, J., Kim, J., and Bae, J. *HiFi-GAN: Generative Adversarial Networks for
   Efficient and High Fidelity Speech Synthesis.* NeurIPS, 2020.
6. McFee, B. et al. *librosa: Audio and Music Signal Analysis in Python.* SciPy,
   2015.
