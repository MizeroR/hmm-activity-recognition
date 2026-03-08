# Human Activity Recognition Using Hidden Markov Models

**Course:** Machine Learning Techniques 2 — Formative 2
**Group 13:** Reine Mizero · Nice Eva Karabaranga

---

## Overview

This project classifies four human activities — **standing, walking, jumping, still** — from smartphone inertial sensor data using a 4-state Gaussian Hidden Markov Model (HMM). The model is trained on 50 labelled sessions recorded across two devices and achieves **92.31% accuracy** on 13 unseen test sessions.

---

## Repository Structure

```
hmm-activity-recognition/
├── data/
│   ├── train/          # 50 labelled sessions (25 per person)
│   └── test/           # 13 unseen sessions (recorded after training)
├── notebooks/
│   └── hmm_activity_recognition.ipynb   # full pipeline
├── outputs/            # saved plots (generated on notebook run)
├── report/             # project report PDF and task sheet PDF
├── requirements.txt
└── README.md
```

Each session folder (e.g. `still_A_01-2026-03-03_20-48-42/`) contains:
- `Accelerometer.csv` — linear acceleration, columns: `time, seconds_elapsed, z, y, x`
- `Gyroscope.csv` — angular velocity, same column layout

---

## Data Collection

| Member | Device | Platform | App | Actual Rate |
|---|---|---|---|---|
| Person A (Nice Eva) | Google Pixel 6 | Android 36 | Sensor Logger 1.54.1 | ~56 Hz |
| Person B (Reine) | iPhone 11 | iOS 26.2.1 | Sensor Logger 1.54 | ~99.5 Hz |

Both devices requested 100 Hz (`sampleRateMs=10`). All sessions are resampled to **100 Hz** via linear interpolation before any feature extraction.

**Training sessions:** 50 total — jumping ×12, standing ×13, still ×13, walking ×12
**Test sessions:** 13 total — jumping ×3, standing ×3, still ×3, walking ×4

---

## Pipeline

### 1. Preprocessing
Sessions are resampled to 100 Hz. A **2-second sliding window** (200 samples) with **50% overlap** (step = 100 samples) is applied, yielding 352 windows total from 50 sessions. An 80/20 stratified session-level split produces 282 training windows (40 sessions) and 70 validation windows (10 sessions).

### 2. Feature Extraction — 23 features per window
- **Time-domain (20):** mean, standard deviation, RMS for each of 6 channels (acc x/y/z + gyr x/y/z); Signal Magnitude Area (SMA); Pearson correlation (acc_x / acc_y)
- **Frequency-domain (3):** dominant frequency (DC excluded), spectral energy, spectral entropy — all from FFT of accelerometer magnitude

All features are Z-score normalised with `StandardScaler` fitted on training data only.

### 3. HMM Training (Baum-Welch)
A single `GaussianHMM` with `covariance_type='full'` and `min_covar=1e-3` is trained using `hmmlearn`. Parameters are initialised from per-activity statistics (means, covariances + `1e-3 × I` regularisation, self-transition probability 0.95). Training converges in **4 iterations** (log-likelihood = 13,682.33, |ΔlogL| = 0.00).

### 4. Viterbi Decoding (Custom NumPy)
A custom log-space Viterbi decoder uses Cholesky decomposition for numerically stable emission probabilities. Validated against hmmlearn's built-in decoder: **100% path agreement**.

---

## Results

| Activity | Test Samples | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|
| Standing | 3 | 66.67% | 100.00% | 92.31% |
| Walking | 4 | 100.00% | 88.89% | 92.31% |
| Jumping | 3 | 100.00% | 100.00% | 100.00% |
| Still | 3 | 100.00% | 100.00% | 100.00% |
| **Overall** | **13** | | | **92.31%** |

Validation accuracy: **100%** (10/10). The single test error is one standing session misclassified as walking, attributable to feature overlap between these two upright activities.

---

## Setup and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook notebooks/hmm_activity_recognition.ipynb
```

Run all cells in order. Plots are saved automatically to `outputs/`.

**Requirements:** `numpy pandas scipy matplotlib seaborn scikit-learn hmmlearn jupyter`

---

## Task Allocation

| Task | Reine Mizero | Nice Eva Karabaranga |
|---|---|---|
| Data collection (still, walking sessions) | ✓ | |
| Data collection (jumping, standing sessions) | | ✓ |
| Feature extraction (time + frequency domain) | ✓ | |
| HMM training & Baum-Welch implementation | | ✓ |
| Custom Viterbi decoder | ✓ | |
| Evaluation metrics & confusion matrices | ✓ | |
| Visualisations | | ✓ |
| Report writing | ✓ | ✓ |
