<div align="center">

![Celesta Title Banner](static/images/title_banner.png)

# Celesta — Kepler Exoplanet Classifier

**India High School Exoplanet Data Challenge**

A leakage-free machine learning system that reads raw Kepler telescope measurements and decides: is this a real planet, a planet candidate, or a false alarm?

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**81.3% accuracy · 78.8% Macro F1 · 78.6% CV F1 · 35 features · 9,564 KOIs**

</div>

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [The Dataset](#2-the-dataset)
3. [How the Model Works](#3-how-the-model-works)
4. [Feature Engineering](#4-feature-engineering)
5. [Results](#5-results)
6. [Run Locally](#6-run-locally)
7. [Run with Docker](#7-run-with-docker)
8. [GitHub Pages Static Demo](#8-github-pages-static-demo)
9. [Web App Features](#9-web-app-features)
10. [API Reference](#10-api-reference)
11. [Project Structure](#11-project-structure)
12. [System Architecture](#12-system-architecture)
13. [Tech Stack](#13-tech-stack)
14. [Credits](#14-credits)

---

## 1. What Problem Are We Solving?

### The Kepler Space Telescope

From 2009 to 2018, NASA's Kepler space telescope stared at a single patch of sky and watched approximately 150,000 stars without blinking. It was looking for one thing: a tiny dip in starlight caused by a planet crossing in front of its star — a **transit**.

When a planet passes between its star and us, the star appears very slightly dimmer. Kepler could detect brightness drops as small as 0.01%. From these tiny dips it found over 9,000 "Objects of Interest" — possible planets that needed investigation.

### The Classification Problem

Not every brightness dip is a planet. Many are caused by:

- **Eclipsing binary stars** — two stars orbiting each other, one blocking the other
- **Background binaries** — a binary star system in the background that happens to align with the target star
- **Instrumental noise** — detector artifacts, cosmic rays, scattered light
- **Grazing transits** — a companion skimming the edge of the star disk

Confirming a planet requires expensive follow-up with ground-based telescopes, radial velocity measurements, and months of analysis. NASA astronomers need a first-pass classifier to prioritise which candidates are worth investigating.

### Three Classes

Every Kepler Object of Interest (KOI) is labelled one of three ways:

| Label | Meaning | Count in dataset |
|---|---|---|
| **CONFIRMED** | Follow-up observations proved it's a real planet | 2,747 |
| **CANDIDATE** | Passes automated checks but awaits confirmation | 1,978 |
| **FALSE POSITIVE** | Shown to not be a planet | 4,839 |

The dataset is **imbalanced** — false positives outnumber candidates 2.4:1, which a naive model would exploit by just predicting FP most of the time. Our model handles this explicitly with balanced class weights.

### What Makes This Hard (and Why Leakage Matters)

NASA's archive includes a field called `koi_pdisposition` — NASA's own automated disposition score. Using it would give us 99%+ accuracy but would be completely useless: we'd just be copying NASA's own answer. That's **data leakage**.

Celesta uses **only raw telescope measurements** — the exact numbers Kepler recorded from the star's light curve and from the star itself. No vetting outputs, no disposition flags, no human-review metadata.

---

## 2. The Dataset

**Source:** [NASA Exoplanet Archive — KOI Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

**Records:** 9,564 Kepler Objects of Interest after cleaning  
**Features used:** 27 raw + 8 engineered = 35 total

### Feature Groups

**Transit shape features** — describe the light curve geometry:

| Feature | Description |
|---|---|
| `koi_period` | Orbital period in days |
| `koi_time0bk` | Time of first transit (BJD - 2454833) |
| `koi_impact` | Impact parameter — how centrally the planet crosses the star disk |
| `koi_duration` | Transit duration in hours |
| `koi_depth` | Transit depth in parts per million — how much the star dims |
| `koi_ror` | Ratio of planet radius to star radius |
| `koi_srho` | Fitted stellar density |
| `koi_prad` | Planet radius in Earth radii |
| `koi_sma` | Semi-major axis in AU |
| `koi_incl` | Orbital inclination in degrees |
| `koi_teq` | Equilibrium temperature in Kelvin |
| `koi_insol` | Insolation flux relative to Earth |
| `koi_dor` | Ratio of semi-major axis to stellar radius |

**Signal statistics** — quality and consistency of the transit signal:

| Feature | Description |
|---|---|
| `koi_max_sngle_ev` | Maximum single-event statistic — strength of the best single transit |
| `koi_max_mult_ev` | Maximum multiple-event statistic — strength across all transits combined |
| `koi_model_snr` | Transit signal-to-noise ratio from the fitted model |
| `koi_count` | Number of times this KOI appears (detrending count) |
| `koi_num_transits` | Number of transits observed |
| `koi_bin_oedp_sig` | Odd-even depth significance — flags eclipsing binaries |

**Stellar properties** — properties of the host star:

| Feature | Description |
|---|---|
| `koi_steff` | Stellar effective temperature in Kelvin |
| `koi_slogg` | Stellar surface gravity (log g) |
| `koi_smet` | Stellar metallicity [Fe/H] |
| `koi_srad` | Stellar radius in solar radii |
| `koi_smass` | Stellar mass in solar masses |

**Position and brightness:**

| Feature | Description |
|---|---|
| `ra` | Right ascension (sky coordinates) |
| `dec` | Declination (sky coordinates) |
| `koi_kepmag` | Kepler magnitude (brightness) |

### Features Deliberately Excluded

These columns exist in the archive but would cause data leakage or are entirely missing:

- `koi_pdisposition` — NASA's automated pre-disposition (leakage)
- `koi_score` — NASA's disposition confidence score (leakage)
- `koi_sage` — stellar age (100% missing in this dataset)
- `koi_model_dof` — model degrees of freedom (100% missing)
- `koi_model_chisq` — chi-squared statistic (100% missing)
- All `koi_vet_*` columns — human vetting outputs (leakage)

---

## 3. How the Model Works

### Why Not Just Use One Model?

No single algorithm dominates on every aspect of this dataset:

- **XGBoost** is excellent at capturing complex non-linear interactions and handles missing values natively
- **HistGradientBoosting** (scikit-learn's gradient boosting) is fast, also handles NaN natively, and responds to class weights
- **Random Forest** is uncorrelated with the boosters — it decorrelates errors and provides a useful diversity signal

A **soft-voting ensemble** averages the probability predictions of all three. When models disagree, the ensemble is more conservative and less likely to be confidently wrong.

### Training Pipeline

```
Raw CSV (9,564 rows × 27 features)
        │
        ▼
Feature Engineering (+8 derived features)
        │
        ▼
Train/test split: 80% / 20%, stratified by class
        │
        ├─► XGBoost (500 trees, balanced sample weights)
        │
        ├─► HistGradientBoosting (500 iterations, class_weight='balanced')
        │
        └─► Random Forest (400 trees, balanced_subsample) + median imputer
                │
                ▼
        Soft Voting (average predicted probabilities)
                │
                ▼
        Final prediction + class probabilities
```

### Handling Class Imbalance

The dataset has 2.4× more FALSE POSITIVEs than CANDIDATEs. Without correction, the model would maximise accuracy by mostly predicting FALSE POSITIVE and ignoring CANDIDATEs.

We fix this with **balanced class weights** on every estimator:
- XGBoost: `compute_sample_weight('balanced', y_train)` — upweights minority class samples
- HistGradientBoosting: `class_weight='balanced'`
- Random Forest: `class_weight='balanced_subsample'`

Each mis-classified CANDIDATE is penalised more heavily than a mis-classified FALSE POSITIVE, forcing the model to pay attention to the harder minority class.

### Cross-Validation

We validate with **5-fold stratified cross-validation** on the full dataset. "Stratified" means each fold has the same class ratio as the full dataset, so no fold accidentally has too many or too few CANDIDATEs.

Result: **78.6% ± 0.7% macro F1** — the low variance (±0.7%) proves the model generalises consistently and isn't just fitting one lucky train/test split.

---

## 4. Feature Engineering

Raw features tell the model what Kepler measured. Engineered features tell it what those measurements *mean* physically. All 8 derived features are computed from the raw measurements — no external data is added.

### 1. `single_multi_ratio` — The False Positive Fingerprint

```python
single_multi_ratio = koi_max_sngle_ev / koi_max_mult_ev
```

**What it means:** The single-event statistic measures how strong the *best single transit* looked. The multi-event statistic measures the *combined* signal across all observed transits. A real planet transits like clockwork — every transit is roughly the same depth, so the multi-event stat grows with the square root of transit count. A background binary or cosmic ray often produces one spectacular event that doesn't repeat — its single-event stat is high but its multi-event stat barely improves.

**In practice:** This engineered feature ranked **#4 in SHAP importance** out of 35 features.

### 2. `duration_period_ratio` — Stellar Density in Disguise

```python
duration_period_ratio = koi_duration / (koi_period × 24)
```

**What it means:** For a circular orbit, transit duration divided by orbital period is proportional to the transit chord length, which depends on the stellar radius and the planet's orbital distance. Through Kepler's Third Law, this ratio encodes the **mean stellar density** — a physical quantity that can be cross-checked against the spectroscopic stellar parameters. Inconsistencies flag false positives.

**In practice:** Ranked **#5 in SHAP importance**.

### 3. Log Transforms: `log_period`, `log_depth`, `log_snr`

```python
log_period = log(1 + koi_period)
log_depth  = log(1 + koi_depth)
log_snr    = log(1 + koi_model_snr)
```

**Why:** Orbital periods range from 0.2 days to 600+ days — three orders of magnitude. Transit depths range from 10 ppm to 100,000+ ppm. These distributions are strongly right-skewed. Tree-based models split on threshold values, so a feature ranging 0–600 has most of its information crammed into the 0–10 range. The log transform spreads this information out and makes decision boundaries easier to find.

### 4. `stellar_density_proxy` — Physical Consistency Check

```python
stellar_density_proxy = koi_smass / koi_srad³
```

**What it means:** This is proportional to the mean density of the host star (M/R³). A planet transiting a dense M-dwarf produces a different light curve shape than one transiting a large, diffuse giant star. Knowing the stellar density helps the model distinguish physically plausible planet scenarios from binary star contamination.

### 5. `impact_ror_ratio` — Grazing Transit Detector

```python
impact_ror_ratio = koi_impact / koi_ror
```

**What it means:** The impact parameter (b) measures how centrally the companion crosses the stellar disk — b=0 means a central crossing, b=1 means just grazing the limb. The radius ratio (Rp/Rs) tells us the size of the transiting object. When b/(Rp/Rs) approaches 1, the companion barely clips the star's edge. Grazing transits are disproportionately common in eclipsing binary false positives because small-but-bright companions at high inclination can mimic planet-depth transits.

### 6. `expected_duration_ratio` — Orbit Shape Flag

```python
expected_duration = (period × 24 / π) × (ror / dor) × √(1 − impact²)
expected_duration_ratio = koi_duration / expected_duration
```

**What it means:** For a circular orbit, we can predict the transit duration purely from geometry. When the observed duration differs significantly from this prediction, something unusual is happening — either an eccentric orbit (the planet speeds up near perihelion, shortening the transit) or a background eclipsing binary with a different geometry than assumed. This ratio flags these anomalies.

---

## 5. Results

### Held-Out Test Set (1,913 KOIs, never seen during training)

| Metric | Value |
|---|---|
| **Accuracy** | **81.3%** |
| **Macro F1** | **78.8%** |
| Weighted F1 | 81.5% |
| Macro Recall | 79.6% |
| Macro Precision | 78.3% |
| **5-fold CV Macro F1** | **78.6% ± 0.7%** |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CANDIDATE | 0.618 | 0.669 | **0.642** | 396 |
| CONFIRMED | 0.845 | 0.887 | **0.866** | 549 |
| FALSE POSITIVE | 0.885 | 0.831 | **0.857** | 968 |

### Confusion Matrix

```
                  Predicted
                  CANDIDATE  CONFIRMED  FALSE POS
Actual CANDIDATE  [  265  ]  [  48   ]  [  83   ]
Actual CONFIRMED  [   41  ]  [  487  ]  [  21   ]
Actual FALSE POS  [  123  ]  [  41   ]  [  804  ]
```

**Reading the matrix:** Numbers on the diagonal are correct predictions. The CANDIDATE row is hardest — by design, CANDIDATE means "we're not sure yet", so the model (and human experts) are genuinely uncertain. The model correctly labels 265 out of 396 candidates (66.9% recall) — significantly better than random guessing (which would get ~21%).

### Improvement Over Baseline

| Model | Macro F1 | Accuracy |
|---|---|---|
| Baseline (25-tree Random Forest, no feature engineering) | 76.8% | 79.2% |
| **Celesta Ensemble** | **78.8%** | **81.3%** |
| Improvement | **+2.0%** | **+2.1%** |

### Top Features by SHAP Importance

SHAP (SHapley Additive exPlanations) measures how much each feature contributed to each prediction, averaged across 500 background samples:

| Rank | Feature | SHAP Score | Type |
|---|---|---|---|
| 1 | `koi_max_mult_ev` | 0.548 | Raw — multi-event statistic |
| 2 | `koi_prad` | 0.339 | Raw — planet radius |
| 3 | `koi_count` | 0.315 | Raw — KOI count |
| 4 | `single_multi_ratio` | 0.264 | **Engineered** |
| 5 | `duration_period_ratio` | 0.203 | **Engineered** |
| 6 | `koi_model_snr` | 0.182 | Raw — transit SNR |
| 7 | `koi_smet` | 0.166 | Raw — stellar metallicity |
| 8 | `koi_bin_oedp_sig` | 0.156 | Raw — odd-even significance |
| 9 | `koi_ror` | 0.148 | Raw — radius ratio |
| 10 | `koi_max_sngle_ev` | 0.136 | Raw — single-event statistic |

Two of the top five features are engineered — proving the feature engineering added real predictive value, not just noise.

---

## 6. Run Locally

**Requirements:** Python 3.9 – 3.11. The pre-trained model is **not** stored in git (it's 49 MB). You need to train it once, or let Docker handle it automatically.

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/celesta.git
cd celesta

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Train the model (takes 3-5 minutes, runs once)
python3 train_model.py

# 4. Start the server
python3 app.py
```

Open **http://localhost:5000**

**Using a virtual environment (recommended):**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
python3 train_model.py
python3 app.py
```

---

## 7. Run with Docker

Docker trains the model **inside the build** — you don't need to run `train_model.py` manually. The Dockerfile is a multi-stage build:

- **Stage 1 (trainer):** installs ML packages, runs `train_model.py`, produces `model.joblib`
- **Stage 2 (runtime):** installs only Flask/gunicorn, copies the trained model from stage 1

```bash
# Build (takes 5-8 minutes on first run — trains the model)
docker build -t celesta .

# Run
docker run -p 5000:5000 celesta

# With Google Custom Search (optional — improves Explore section images)
docker run -p 5000:5000 \
  -e GOOGLE_CSE_CX=your_cx_id \
  -e GOOGLE_CSE_API_KEY=your_api_key \
  celesta
```

Open **http://localhost:5000**

**Why multi-stage?** The ML training packages (XGBoost, SHAP, imbalanced-learn) add ~800 MB to the image. By training in stage 1 and only copying the output (`model.joblib`, `stats.json`) to stage 2, the final runtime image stays as lean as possible.

---

## 8. GitHub Pages Static Demo

The `docs/` folder is a fully static version of the site — no Python, no Flask, no server required. It shows the same 3D hero, the Explore section, and the Performance dashboard with metrics baked in. The prediction form shows a friendly note that the live backend is needed for actual predictions.

**Enable GitHub Pages:**

1. Push this repo to GitHub
2. Go to repo **Settings → Pages**
3. Source: **Deploy from a branch**
4. Branch: `main`, Folder: `/docs`
5. Save — your site goes live at `https://username.github.io/celesta/`

The static site assets include cache-busting query strings (`?v=2`) so GitHub's CDN (Fastly) always serves fresh files after each push.

---

## 9. Web App Features

### Overview Section
Live stats pulled from `data/stats.json` — accuracy, macro F1, total KOIs, number of classes. Animated number counters on scroll.

### Explore the Universe
Search any star, exoplanet, nebula, or galaxy. Results pull from:
- **Wikipedia REST API** — article extract and thumbnail
- **NASA Images API** — high-quality telescope photography
- **Google Custom Search** (optional) — broader image coverage

Includes 11 featured objects (Orion Nebula, Andromeda Galaxy, Betelgeuse, etc.) with curated NASA/ESO images.

### Performance Dashboard
- Confusion matrix (3×3 grid)
- Per-class precision, recall, F1
- SHAP-ranked feature importance bar chart
- Cross-validation results

### Live Prediction
Fill in any combination of the 35 Kepler features, click Classify, and get:
- Predicted disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE)
- Confidence score
- Full probability distribution across all three classes
- List of any missing fields (the model handles them with NaN)

### Developer Section
Methodology, feature descriptions, and acknowledgements.

---

## 10. API Reference

All endpoints return JSON. The server sets `Cache-Control: no-cache` on every response.

### `GET /api/stats`

Returns the full model metrics object from `data/stats.json`.

```bash
curl http://localhost:5000/api/stats
```

```json
{
  "accuracy": 0.8134,
  "macro_f1": 0.7884,
  "cv_macro_f1_mean": 0.7865,
  "cv_macro_f1_std": 0.0071,
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
  "feature_importance": [["koi_max_mult_ev", 0.548], ...]
}
```

### `POST /api/predict`

Send any subset of the 35 features. Missing values are handled gracefully (gradient boosted models accept NaN natively; Random Forest uses median imputation).

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "koi_period": 2.204,
    "koi_prad": 1.5,
    "koi_model_snr": 45.3,
    "koi_max_mult_ev": 30.1,
    "koi_impact": 0.3,
    "koi_duration": 2.1,
    "koi_depth": 800,
    "koi_ror": 0.04,
    "koi_dor": 12.5,
    "koi_smass": 1.0,
    "koi_srad": 1.0
  }'
```

```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.8731,
  "probabilities": {
    "CANDIDATE": 0.0512,
    "CONFIRMED": 0.8731,
    "FALSE POSITIVE": 0.0757
  },
  "missing_fields": ["koi_time0bk", "koi_srho", "koi_sma", "..."]
}
```

### `GET /api/explore/search?q=orion`

Wikipedia open search — returns up to 8 matching article titles and descriptions.

### `GET /api/explore/details?name=Orion+Nebula`

Full object details: Wikipedia extract, description, image URLs (NASA priority → Wikipedia → Google), and Wikipedia article link.

---

## 11. Project Structure

```
celesta/
├── app.py                     # Flask server — routes, prediction endpoint, Explore API
├── train_model.py             # Training script — loads CSV, engineers features, trains ensemble, saves model
├── model_utils.py             # BalancedXGBClassifier — shared between train and app for joblib compatibility
├── requirements.txt           # All Python dependencies
├── Dockerfile                 # Multi-stage build: train model in stage 1, serve in stage 2
├── .dockerignore              # Keeps Docker build context small
├── .gitignore                 # Excludes model.joblib (49 MB), raw CSV, __pycache__
├── Celesta_Colab.ipynb        # Self-contained Google Colab notebook — downloads data, trains, evaluates
├── Celesta_Hackathon_Notebook.ipynb  # Local notebook (requires koi_stripped.csv)
├── README.md                  # This file
│
├── data/
│   ├── koi_stripped.csv       # Cleaned training CSV (1.86 MB) — needed for train_model.py and Docker
│   └── stats.json             # Pre-computed metrics served to the frontend
│
├── static/
│   ├── css/style.css          # All styles (dark space theme, responsive)
│   ├── js/app.js              # Three.js hero planet, Explore section, prediction form
│   ├── model.json             # Feature definitions for the prediction form
│   └── images/                # Textures (planet, starfield, thumbnails), favicon, developer photo
│
├── templates/
│   └── index.html             # Jinja2 template rendered by Flask
│
└── docs/                      # Static GitHub Pages snapshot (no Flask required)
    ├── index.html             # Same UI with stats baked in
    └── static/                # Copy of CSS, JS, images for Pages deployment
```

---

## 12. System Architecture

Celesta is split into three completely independent runtimes that share only the trained model file. You can run any one of them without the others.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CELESTA SYSTEM                               │
│                                                                     │
│  ┌─────────────────────────┐   ┌───────────────────────────────┐   │
│  │   TRAINING PIPELINE     │   │      FLASK WEB SERVER         │   │
│  │   (train_model.py)      │   │      (app.py)                 │   │
│  │                         │   │                               │   │
│  │  koi_stripped.csv       │   │  GET  /          → HTML       │   │
│  │       │                 │   │  GET  /api/stats → JSON       │   │
│  │       ▼                 │   │  POST /api/predict → JSON     │   │
│  │  Feature Engineering    │   │  GET  /api/explore/search     │   │
│  │  (27 raw → 35 total)    │   │  GET  /api/explore/details    │   │
│  │       │                 │   │        │                      │   │
│  │       ▼                 │   │        ▼                      │   │
│  │  VotingClassifier       │──►│  model.joblib  (49 MB)        │   │
│  │  ├ XGBoost              │   │  data/stats.json              │   │
│  │  ├ HistGradBoost        │   │                               │   │
│  │  └ RandomForest         │   │  Wikipedia REST API  ──────►  │   │
│  │       │                 │   │  NASA Images API    ──────►   │   │
│  │       ▼                 │   │  Google CSE (opt.)  ──────►   │   │
│  │  model.joblib           │   └───────────────────────────────┘   │
│  │  data/stats.json        │                                        │
│  └─────────────────────────┘   ┌───────────────────────────────┐   │
│                                 │   GITHUB PAGES STATIC DEMO   │   │
│                                 │   (docs/index.html)          │   │
│                                 │                               │   │
│                                 │   No Python. No server.       │   │
│                                 │   Stats baked in at build.    │   │
│                                 │   Prediction form shows note  │   │
│                                 │   that live backend needed.   │   │
│                                 └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.1 Training Pipeline

The training script (`train_model.py`) runs once offline and produces two artefacts consumed by the web server.

```
koi_stripped.csv  (1.86 MB — 9,564 rows × 27 raw feature columns + target)
        │
        ▼
   add_features()
   Computes 8 derived columns from the 27 raw ones.
   All arithmetic uses eps = 1e-9 in denominators to survive edge-case rows.
   Result: DataFrame with 35 feature columns.
        │
        ├──────────────────────────────────────────────────────────────
        │  80/20 stratified split → (X_train, y_train) / (X_test, y_test)
        │──────────────────────────────────────────────────────────────
        │
        ├─► BalancedXGBClassifier            (model_utils.py)
        │     XGBoost 2.x, n_estimators=500, max_depth=6, lr=0.04
        │     subsample=0.8, colsample_bytree=0.8
        │     reg_alpha=0.1, reg_lambda=1.0
        │     tree_method="hist" (handles NaN natively)
        │     sample_weight = compute_sample_weight("balanced", y_train)
        │
        ├─► HistGradientBoostingClassifier   (scikit-learn)
        │     max_iter=500, max_depth=8, lr=0.04
        │     min_samples_leaf=20, l2_regularization=0.1
        │     class_weight="balanced"
        │     Handles NaN natively
        │
        └─► Pipeline([SimpleImputer(median), RandomForestClassifier])
              RF: n_estimators=400, max_depth=16, max_features="sqrt"
              class_weight="balanced_subsample"
              Imputer needed because RF cannot handle NaN directly
                        │
                        ▼
              VotingClassifier(voting="soft")
              Averages probability vectors from all three estimators.
              Final class = argmax of averaged probabilities.
                        │
                        ├─► 5-fold stratified CV → cv_macro_f1 reported
                        ├─► Held-out test evaluation → stats.json
                        └─► SHAP TreeExplainer on XGBoost sub-model
                                        │
                                        ▼
                            model.joblib     (joblib-pickled dict)
                            data/stats.json  (metrics + feature importance)
```

**model.joblib internal layout:**

```python
{
    "pipeline":      VotingClassifier,   # the fitted ensemble
    "label_encoder": LabelEncoder,       # int → "CONFIRMED" / "CANDIDATE" / "FALSE POSITIVE"
    "features":      list[str],          # 35 names, in column order expected by pipeline
    "engineered":    list[str],          # 8 engineered names — tells app.py which to compute
}
```

### 12.2 Web Server — Request / Response Flow

```
Browser
   │
   │  POST /api/predict   {"koi_period": 2.2, "koi_prad": 1.5, ...}
   ▼
Flask app.py
   │
   ├─ parse JSON body
   ├─ for each of the 27 raw features:
   │     if present → cast to float
   │     if missing / empty → NaN   (recorded in missing_fields list)
   │
   ├─ _add_features(df)
   │     Same 8 formulas as train_model.py
   │     Applied to the single-row DataFrame
   │
   ├─ X = df[features]   (35 columns, ordered)
   │
   ├─ VotingClassifier.predict(X)       → int index
   ├─ VotingClassifier.predict_proba(X) → [p_CANDIDATE, p_CONFIRMED, p_FP]
   ├─ LabelEncoder.classes_[index]      → string label
   │
   └─ return JSON
         {
           "prediction":    "CONFIRMED",
           "confidence":    0.8731,
           "probabilities": {"CANDIDATE": 0.051, "CONFIRMED": 0.873, "FALSE POSITIVE": 0.076},
           "missing_fields": ["koi_srho", "koi_time0bk", ...]
         }
```

The model is loaded once on first request and cached in `_model_cache`. Every subsequent prediction reuses the in-memory object — no disk I/O per request.

### 12.3 Explore Section — External API Chain

The Explore section uses a three-tier fallback for images to maximise quality and coverage:

```
User types a search query
        │
        ▼
GET /api/explore/search?q=...
        │
        └─► Wikipedia OpenSearch API (free, no key)
              Returns up to 8 {title, description} results
              Cached with lru_cache(maxsize=128)

User selects a result
        │
        ▼
GET /api/explore/details?name=...
        │
        ├─► Wikipedia REST API /page/summary/{slug}
        │     Returns: title, description, extract, thumbnail, originalimage, wiki_link
        │     Cached with lru_cache(maxsize=128)
        │
        ├─► NASA Images API /search?q=...&media_type=image&page_size=10
        │     Filters out noise titles (launches, crew portraits, etc.)
        │     using _nasa_relevant() keyword blacklist
        │     Returns first image that passes the filter
        │     Cached with lru_cache(maxsize=128)
        │
        └─► Google Custom Search API (optional)
              Only called if GOOGLE_CSE_CX + GOOGLE_CSE_API_KEY are set
              Searches for "{name} space telescope" with searchType=image
              100 queries/day free tier
              Silently skipped if credentials absent
                        │
                        ▼
              Final image priority: NASA > Wikipedia full-size > Google
              Response includes up to 3 image URLs (all three sources)
```

### 12.4 Docker Multi-Stage Build

```
docker build -t celesta .
        │
        ▼
┌───────────────────────────────────────────────┐
│  Stage 1: trainer  (python:3.11-slim)         │
│                                               │
│  apt install build-essential                  │
│  pip install -r requirements.txt              │
│    (includes xgboost, shap, imbalanced-learn) │
│                                               │
│  COPY train_model.py model_utils.py           │
│  COPY data/koi_stripped.csv                   │
│                                               │
│  RUN python3 train_model.py                   │
│    → /build/model.joblib    (49 MB)           │
│    → /build/data/stats.json                   │
└───────────────────────────────────────────────┘
        │ COPY --from=trainer
        ▼
┌───────────────────────────────────────────────┐
│  Stage 2: runtime  (python:3.11-slim)         │
│                                               │
│  pip install flask gunicorn requests numpy    │
│    (≈ 80 MB — no xgboost/shap at runtime)    │
│                                               │
│  COPY app.py model_utils.py                   │
│  COPY templates/ static/                      │
│  COPY model.joblib       ← from stage 1       │
│  COPY data/stats.json    ← from stage 1       │
│                                               │
│  EXPOSE 5000                                  │
│  CMD gunicorn --bind 0.0.0.0:5000             │
│              --workers 2 --timeout 120        │
│              app:app                          │
└───────────────────────────────────────────────┘
```

The training packages (XGBoost, SHAP, imbalanced-learn, scikit-learn extras) add roughly 800 MB to stage 1 but are completely absent from the final runtime image. Only the trained `model.joblib` binary crosses the stage boundary.

### 12.5 Frontend Architecture

```
templates/index.html   (Jinja2 — rendered by Flask at request time)
        │
        ├── static/css/style.css
        │     Custom properties (CSS variables) for the dark space palette.
        │     Responsive layout via CSS Grid and Flexbox — no framework.
        │     Sections: hero, overview, explore, performance, classifier, developer.
        │
        └── static/js/app.js   (vanilla JS — no bundler, no build step)
              │
              ├── Three.js WebGL scene (lazy-init on requestIdleCallback)
              │     ├── 1 200 star particles (BufferGeometry + PointsMaterial)
              │     ├── Milky Way skybox (SphereGeometry, BackSide texture)
              │     ├── Hero planet (SphereGeometry, texture-mapped)
              │     ├── 3 ambient glow orbs (low-opacity MeshBasicMaterial)
              │     └── Animation loop: planet rotation, mouse parallax
              │
              ├── Explore section
              │     ├── Debounced input → fetch /api/explore/search
              │     ├── Click on result → fetch /api/explore/details
              │     └── Renders Wikipedia extract + image carousel
              │
              ├── Performance dashboard
              │     ├── Confusion matrix rendered from stats.json data
              │     ├── Animated number counters (IntersectionObserver)
              │     └── SHAP bar chart (pure CSS width transitions)
              │
              └── Classifier form
                    ├── 35 feature inputs (27 raw visible, 8 engineered hidden)
                    ├── fetch POST /api/predict
                    └── Renders label badge + probability bar chart
```

All frontend logic is in a single file with no build step. The Three.js scene is initialised lazily using `requestIdleCallback` so it never blocks the initial page render.

### 12.6 Cache Strategy

| Layer | Mechanism | TTL |
|---|---|---|
| Browser / proxy | `Cache-Control: no-cache, no-store, must-revalidate` | 0 s — always revalidate |
| Static files (Flask) | `SEND_FILE_MAX_AGE_DEFAULT = 0` | 0 s |
| GitHub Pages CSS/JS | `?v=2` query string cache-buster | Until version bump |
| Wikipedia API results | `lru_cache(maxsize=128)` per process | Process lifetime |
| NASA image results | `lru_cache(maxsize=128)` per process | Process lifetime |
| Model artefacts | Module-level `_model_cache` global | Process lifetime |
| Prediction responses | No caching — always live | 0 s |

---

## 13. Tech Stack

### Machine Learning

| Library | Version | Role | Why chosen |
|---|---|---|---|
| **XGBoost** | ≥ 2.0 | Primary ensemble member | Gold standard for tabular classification. Second-order gradient optimisation finds complex non-linear boundaries. Native NaN handling via `tree_method="hist"` avoids imputation for the majority of features. Parallelises across all CPU cores (`n_jobs=-1`). |
| **scikit-learn** | 1.9 | Ensemble framework + RF + HGB + utilities | `VotingClassifier` wires the ensemble together. `HistGradientBoostingClassifier` is scikit-learn's own gradient booster — it handles NaN natively, respects `class_weight="balanced"`, and trains faster than XGBoost on smaller datasets. `RandomForestClassifier` provides a high-variance, bagged estimator whose errors are largely uncorrelated with the two boosters. `LabelEncoder`, `SimpleImputer`, `Pipeline`, `StratifiedKFold`, `cross_val_score`, and all metric functions also come from scikit-learn. |
| **imbalanced-learn** | ≥ 0.12 | Class weight computation | `compute_sample_weight("balanced", y)` from `sklearn.utils.class_weight` is standard scikit-learn, but imbalanced-learn is listed as a dependency for its complementary tooling. |
| **SHAP** | ≥ 0.44 | Feature explainability | `TreeExplainer` computes exact Shapley values for tree ensembles in polynomial time. Used on the XGBoost sub-model to rank all 35 features by mean |SHAP| across 500 background samples and three classes. Results are written to `stats.json` and displayed in the performance dashboard. |
| **pandas** | 3.x | Data loading and transformation | Reads the CSV, applies feature engineering column-by-column, and produces the `DataFrame` consumed by the pipeline. Used at both training time and inference time (single-row DataFrame for prediction). |
| **numpy** | 2.x | Numerical operations | Log transforms, clipping, `sqrt`, epsilon guards, and array manipulation throughout feature engineering and SHAP computation. |
| **joblib** | 1.x | Model serialisation | Saves and loads `model.joblib`. joblib is preferred over pickle for scikit-learn objects because it handles large numpy arrays efficiently with memory-mapped files. |

### BalancedXGBClassifier — Custom Wrapper

scikit-learn 1.9 tightened metadata routing: passing `sample_weight` through `VotingClassifier.fit()` now raises a `ValueError` unless every estimator explicitly declares it supports the parameter. XGBoost does not do this.

`BalancedXGBClassifier` (`model_utils.py`) solves the problem by computing `compute_sample_weight("balanced", y)` inside its own `fit()` method before delegating to `XGBClassifier.fit()`. The caller — `VotingClassifier` — never needs to pass `sample_weight` at all.

The class lives in `model_utils.py` (not in `train_model.py` or `app.py`) because joblib resolves class references by module path when unpickling. Both scripts import from the same module, so the unpickling succeeds in both contexts.

### Backend

| Library | Version | Role | Why chosen |
|---|---|---|---|
| **Flask** | 3.1 | Web framework | Minimal surface area — the app has 5 routes and no ORM, session management, or authentication. Flask adds ~2 MB to the runtime image versus 50+ MB for Django. Jinja2 templating is bundled. |
| **gunicorn** | 23.x | WSGI server (Docker/production) | Multi-worker pre-fork server. Configured with 2 workers and a 120 s timeout (model load on a cold start takes ~3 s). Flask's built-in development server is single-threaded and not suitable for concurrent requests. |
| **requests** | latest | HTTP client for external APIs | Used by all three Explore API helpers (`_wiki_search`, `_wiki_summary`, `_nasa_image`, `_google_image`). A 6-second timeout guards against slow upstream responses. |

### Frontend

| Technology | Role | Detail |
|---|---|---|
| **Three.js r134** | 3D WebGL scene | Loaded from jsDelivr CDN. Renders 1 200 star particles, a Milky Way skybox sphere (texture mapped, `BackSide`), a rotating hero planet, and 3 low-opacity ambient glow orbs. Mouse position drives a parallax offset on the camera. Initialised lazily using `requestIdleCallback` to avoid blocking the initial DOM paint. |
| **Vanilla JavaScript (ES2020)** | Application logic | No framework, no bundler. A single `app.js` file handles Three.js, the Explore search UI, the performance dashboard, the classifier form, animated counters, and IntersectionObserver-triggered scroll animations. |
| **CSS custom properties** | Design system | All colours, spacings, and radii are CSS variables defined on `:root`. The dark space palette (`#0a0b14` background, `#6366f1` accent) is applied consistently via these variables — changing the theme requires editing ~15 lines. |
| **CSS Grid + Flexbox** | Layout | Responsive layout with no CSS framework. The performance dashboard uses a 3×3 CSS Grid for the confusion matrix. Feature input cards use `auto-fill` grid columns. |
| **Inter (Google Fonts)** | Typography | Variable-weight sans-serif. Loaded from Google Fonts CDN with `font-display: swap` to prevent render-blocking. |

### External APIs

| API | Auth | Rate limit | Used for |
|---|---|---|---|
| **Wikipedia OpenSearch** | None | Generous (no documented limit) | Explore search autocomplete — `GET /api/explore/search` |
| **Wikipedia REST /page/summary** | None | Generous | Explore details — article extract, description, thumbnail, full image |
| **NASA Images API** | None | No documented limit | Explore details — high-quality telescope photography, filtered to remove off-topic results |
| **Google Custom Search API** | API key + CX required | 100 queries/day free | Explore details — image fallback when NASA returns nothing relevant. Entirely optional; silently skipped if credentials are absent. |

### Deployment

| Technology | Role | Detail |
|---|---|---|
| **Docker** | Containerised build and run | Multi-stage Dockerfile. Stage 1 (`trainer`, `python:3.11-slim`) installs all ML packages and runs `train_model.py`. Stage 2 (`runtime`, `python:3.11-slim`) installs only Flask and gunicorn, copies `model.joblib` and `stats.json` from stage 1. The final image is ~500 MB lighter than a single-stage build. |
| **GitHub Pages** | Static demo hosting | The `docs/` directory is served directly by GitHub Pages from the `main` branch. No server needed. Stats are baked into the HTML at development time. Cache-busting `?v=2` query strings on CSS/JS links and `<meta http-equiv="Cache-Control">` tags prevent Fastly CDN from serving stale assets. |
| **Replit** | Development environment | NixOS-based sandbox. LightGBM was excluded from the ensemble because `libgomp.so.1` (OpenMP) is not present in the Replit NixOS environment — XGBoost and HistGradientBoosting provide equivalent coverage without it. |

### Python Version

Python **3.11** throughout — chosen for XGBoost 2.x compatibility, scikit-learn 1.9 support, and `match` statement availability if needed in future extensions. The `python:3.11-slim` Docker base image keeps the runtime image as small as possible.

---

## 14. Credits

- **Dataset:** [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Kepler Objects of Interest Cumulative Table (public domain)
- **Challenge:** India High School Exoplanet Data Challenge / Celesta
- **Planet textures:** [Solar System Scope](https://www.solarsystemscope.com/textures/) (CC BY 4.0)
- **Live object data:** Wikipedia REST API, NASA Images API (both free, no key required)
- **SHAP library:** Lundberg & Lee, 2017 — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

---

<div align="center">

Made by **Srinath V Venkateshan**

[LinkedIn](https://www.linkedin.com/in/srinath-v-a26b372b7/) · [vvsrinath0@gmail.com](mailto:vvsrinath0@gmail.com) · [GitHub](https://github.com/vvsrinath)

</div>
