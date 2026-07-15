<div align="center">

<img src="static/images/title_banner.png" alt="Celesta" width="100%"/>

# 🌌 Celesta — Kepler Exoplanet Classifier

### India High School Exoplanet Data Challenge

*Leakage-free machine learning that reads raw Kepler telescope measurements<br>and decides: real planet, planet candidate, or false alarm.*

<br>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-E76F00?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue?style=for-the-badge&logo=github)](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)

<br>

[Source Code](https://github.com/vvsrinath/Celesta-India-High-School-Exoplanet-Data-Challenge) | [Live Demo](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)

<br>

| 🎯 Accuracy | 📊 Macro F1 | 🔁 CV F1 | 🌠 Features | 🪐 KOIs |
|:-----------:|:-----------:|:--------:|:-----------:|:-------:|
| **81.3 %** | **78.8 %** | **78.6 % ± 0.7 %** | **39** | **9,564** |

<br>

**Developed by Srinath Vatchavari Venkateshan**

</div>

---

## Table of Contents

1. [The Problem — What Are We Classifying?](#1-the-problem)
2. [Why Leakage-Free Matters](#2-why-leakage-free-matters)
3. [The Dataset](#3-the-dataset)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Handling Missing Values & Class Imbalance](#5-handling-missing-values--class-imbalance)
6. [Feature Engineering — 12 Physics-Motivated Features](#6-feature-engineering)
7. [Model Architecture — The Ensemble](#7-model-architecture)
8. [Results](#8-results)
9. [SHAP Explainability — What the Model Learned](#9-shap-explainability)
10. [Live Web Application](#10-live-web-application)
11. [Run It Yourself](#11-run-it-yourself)
12. [Project Structure](#12-project-structure)
13. [API Reference](#13-api-reference)
14. [Tech Stack](#14-tech-stack)
15. [Credits](#15-credits)

---

## 1. The Problem

### The Kepler Space Telescope

From 2009 to 2018, NASA's Kepler telescope stared at a single patch of sky and watched **~150,000 stars** without blinking. It was hunting for one thing: the faint, rhythmic dimming that betrays a planet crossing in front of its star — a **transit**.

Kepler could detect brightness drops as small as **0.01%** — roughly equivalent to a mosquito crossing a car's headlight from a mile away.

Every transit-like signal gets logged as a **Kepler Object of Interest (KOI)** and given one of three dispositions:

| Label | What it means | Count |
|---|---|---|
| 🟢 **CONFIRMED** | Follow-up spectroscopy proved it's a real planet | 2,747 |
| 🟡 **CANDIDATE** | Passed automated screening — awaits confirmation | 1,978 |
| 🔴 **FALSE POSITIVE** | Follow-up proved it is *not* a planet | 4,839 |

### Why This Is Hard

Not every dip in starlight is a planet. Imposters include:

- **Eclipsing binary stars** — two stars orbiting each other, one blocking the other
- **Background eclipsing binaries** — a binary star sitting behind the target star
- **Instrumental noise** — cosmic rays, detector crosstalk, scattered light
- **Grazing transits** — a stellar companion just skimming the star's limb

Confirming a planet requires expensive ground-based follow-up: radial velocity measurements, transit timing variations, and months of analysis. A reliable first-pass classifier saves enormous telescope time by flagging the best candidates and filtering out noise.

---

## 2. Why Leakage-Free Matters

The NASA archive includes two columns that give away the answer:

| Column | What it is | Why we exclude it |
|---|---|---|
| `koi_pdisposition` | NASA's own pre-classification | Direct leakage — we'd be copying NASA's answer |
| `koi_score` | NASA's confidence in that classification | Same |
| `koi_vet_*` | Human vetting flags | Same |

Using these columns produces 99%+ accuracy but is **scientifically worthless**. The model would learn nothing about the underlying astrophysics — it would just be a lookup table for NASA's opinion.

> **Celesta's constraint:** every input to the model is a raw physical measurement from the Kepler photometer or a stellar spectroscopy catalogue — the exact same data an astrophysicist would start with before touching a telescope.

---

## 3. The Dataset

**Source:** [NASA Exoplanet Archive — KOI Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)  
**Records:** 9,564 KOIs after cleaning  
**Raw features used:** 27  
**Engineered features:** 12  
**Total features:** 39

### Feature Groups

**Transit shape** — describes the light curve geometry:

| Feature | Description |
|---|---|
| `koi_period` | Orbital period in days |
| `koi_time0bk` | Time of first transit (BJD − 2454833) |
| `koi_impact` | Impact parameter — 0 = central crossing, 1 = grazing |
| `koi_duration` | Transit duration in hours |
| `koi_depth` | Transit depth in parts per million |
| `koi_ror` | Planet radius / star radius |
| `koi_srho` | Fitted stellar density from transit |
| `koi_prad` | Planet radius in Earth radii |
| `koi_sma` | Semi-major axis in AU |
| `koi_incl` | Orbital inclination in degrees |
| `koi_teq` | Equilibrium temperature in Kelvin |
| `koi_insol` | Insolation flux relative to Earth |
| `koi_dor` | Semi-major axis / stellar radius ratio |

**Signal statistics** — quality and consistency of the transit:

| Feature | Description |
|---|---|
| `koi_max_sngle_ev` | Best single-transit statistic |
| `koi_max_mult_ev` | Combined multi-transit statistic |
| `koi_model_snr` | Transit model SNR |
| `koi_count` | Number of times this KOI was detected |
| `koi_num_transits` | Observed transit count |
| `koi_bin_oedp_sig` | Odd-even depth significance — the primary eclipsing binary flag |

**Stellar properties:**

| Feature | Description |
|---|---|
| `koi_steff` | Stellar effective temperature in Kelvin |
| `koi_slogg` | Stellar surface gravity (log g) |
| `koi_smet` | Stellar metallicity [Fe/H] |
| `koi_srad` | Stellar radius in solar radii |
| `koi_smass` | Stellar mass in solar masses |

**Position and brightness:** `ra`, `dec`, `koi_kepmag`

---

## 4. Exploratory Data Analysis

Before building any model, we ran a thorough EDA to understand the data's structure, distributions, and class-separating signals.

### 4.1 Class Imbalance

```
FALSE POSITIVE  ████████████████████████  4,839  (50.6%)
CONFIRMED       ██████████████            2,747  (28.7%)
CANDIDATE       ██████████                1,978  (20.7%)
```

FALSE POSITIVE outnumbers CANDIDATE by **2.4×**. A naive model that always predicts FALSE POSITIVE achieves ~50% accuracy while learning nothing useful about real planets. We handle this explicitly — see Section 5.

### 4.2 Key Observations from Feature Distributions

**`koi_max_mult_ev` (Multi-Event Statistic)**  
The single most discriminating raw feature. Confirmed planets show a tall, narrow peak at high values — their signal repeats consistently. False positives spread across a wide, low range — their signal doesn't stack the way a repeating transit does.

**`koi_prad` (Planet Radius)**  
Confirmed planets cluster under 10 Earth radii. False positives spread widely — because eclipsing binaries, which mimic large transit depths, produce apparent "planet" radii far exceeding what any plausible planet could have.

**`koi_impact` (Impact Parameter)**  
Values near 1.0 mean the companion barely grazes the stellar limb. This configuration is far more common in eclipsing binary false positives than in genuine planets, which tend to have more central transits.

**`koi_bin_oedp_sig` (Odd-Even Depth Significance)**  
When a binary star system is seen edge-on, odd-numbered transits (star A blocking star B) differ in depth from even-numbered transits (star B blocking star A). This alternating pattern is rare in real planets and is the clearest single-feature indicator of an eclipsing binary.

**`koi_smet` (Stellar Metallicity)**  
A scientifically meaningful separation: confirmed planet hosts are systematically more metal-rich, consistent with the well-established correlation between stellar metallicity and planet formation efficiency.

### 4.3 Missingness Profile

```
koi_bin_oedp_sig   ████████████████  15.8%   ← handle with care
koi_max_mult_ev    ████████████      11.9%   ← handle with care
koi_max_sngle_ev   ████████████      11.9%   ← handle with care
koi_num_transits   ████████████      11.9%
koi_smet           ███               4.0%
koi_teq            ██                3.8%
(all others)       █                 <4.0%
```

No features are dropped. Missing values are handled in-model (see Section 5).

---

## 5. Handling Missing Values & Class Imbalance

### Missing Values

We never impute before the train/test split — that would leak test-set statistics into training.

| Model | Strategy |
|---|---|
| **XGBoost** | Native NaN handling — learns the optimal branch for missing values at each split |
| **HistGradientBoosting** | Native NaN handling — same mechanism |
| **Random Forest** | `SimpleImputer(strategy='median')` inside a scikit-learn `Pipeline`, fitted only on training data |

### Class Imbalance

We use **balanced sample weights** on every estimator — no synthetic oversampling, which can introduce artefacts in high-dimensional data with many missing values.

| Model | Mechanism |
|---|---|
| XGBoost | `compute_sample_weight('balanced', y_train)` injected via custom subclass |
| HistGradientBoosting | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced_subsample'` — rebalances each bootstrap draw |

Each mis-classified CANDIDATE is penalised proportionally more than a mis-classified FALSE POSITIVE, forcing the model to attend to the harder minority class.

---

## 6. Feature Engineering

Raw Kepler measurements tell the model *what* was recorded. Engineered features tell it *what those measurements mean physically*. All 12 are derived from the raw columns — no external data added.

### Original 8 — Celesta v1

| # | Feature | Formula | Physical intuition |
|---|---|---|---|
| 1 | `single_multi_ratio` | SNG ÷ MULT | High → one spectacular event that didn't repeat — false positive fingerprint |
| 2 | `duration_period_ratio` | dur ÷ (P × 24) | Encodes transit chord length; tied to stellar density via Kepler's third law |
| 3 | `log_period` | log(1 + P) | Compresses three-order-of-magnitude range for tree splits |
| 4 | `log_depth` | log(1 + depth) | Same — transit depth is heavily right-skewed |
| 5 | `log_snr` | log(1 + SNR) | Same |
| 6 | `stellar_density_proxy` | M★ ÷ R★³ | Proportional to mean stellar density; validates transit geometry |
| 7 | `impact_ror_ratio` | b ÷ (Rp/R★) | Near 1 → grazing transit → common in eclipsing binary false positives |
| 8 | `expected_duration_ratio` | T_obs ÷ T_Keplerian | Deviations signal orbital eccentricity or a blended background binary |

### New 4 — Celesta v2 (added for this submission)

| # | Feature | Formula | Physical intuition |
|---|---|---|---|
| 9 | `snr_per_transit` | SNR ÷ √N_transits | Per-transit SNR — consistent for real planets; noisy for one-off artefacts |
| 10 | `odd_even_norm` | odd-even sig ÷ SNR | Normalised eclipsing binary flag — high means depth alternates between transits |
| 11 | `log_srad` | log(1 + R★) | Stellar radius spans 0.1 – 10 R☉; log-compression improves tree splits |
| 12 | `depth_ror_check` | depth ÷ (ror² × 10⁶) | Geometric consistency: for a clean circular transit this ratio should ≈ 1 |

> **Result:** 2 of the top 5 SHAP-ranked features are engineered, proving the feature engineering added real predictive value rather than noise.

---

## 7. Model Architecture

### Why a Soft-Voting Ensemble?

No single algorithm dominates every aspect of this dataset:

| Model | Strength | Limitation |
|---|---|---|
| **XGBoost** | Captures complex non-linear interactions; native NaN; fastest boosting | Needs explicit class-weight injection |
| **HistGradientBoosting** | Fast; honours `class_weight`; native NaN; strong regularisation | Slightly less expressive than XGBoost |
| **Random Forest** | Decorrelated errors; stable on small classes; high diversity | Requires median imputation for NaN |

A **soft-voting ensemble** averages the *probability* predictions of all three. When models disagree, the ensemble is more conservative and less likely to be confidently wrong. This is the core reason the ensemble outperforms every individual model.

### Training Pipeline

```
Raw CSV  (9,564 rows × 27 features)
        │
        ▼
Feature Engineering  (+12 derived columns → 39 total)
        │
        ▼
Stratified 80/20 split  (preserves class ratios in train and test)
        │
        ├──► BalancedXGBClassifier
        │      500 trees · lr=0.04 · max_depth=6 · subsample=0.8
        │      compute_sample_weight('balanced') injected automatically
        │
        ├──► HistGradientBoostingClassifier
        │      500 iterations · lr=0.04 · max_depth=8
        │      class_weight='balanced'
        │
        └──► Pipeline[ SimpleImputer(median) → RandomForestClassifier ]
               400 trees · max_depth=16 · class_weight='balanced_subsample'
                        │
                        ▼
              VotingClassifier(voting='soft')
              Averages probabilities → argmax → final label
                        │
                        ├──► 5-fold stratified CV  →  78.6% ± 0.7% Macro F1
                        ├──► Held-out test set     →  81.3% accuracy · 78.8% Macro F1
                        └──► SHAP TreeExplainer    →  feature importance
```

### BalancedXGBClassifier

XGBoost's `class_weight` parameter is not honoured the way scikit-learn handles it. Rather than passing sample weights through `VotingClassifier.fit()` (which fails in scikit-learn 1.9 due to metadata routing), a lightweight subclass computes and injects balanced weights automatically:

```python
class BalancedXGBClassifier(XGBClassifier):
    def fit(self, X, y, **kwargs):
        kwargs.setdefault('sample_weight', compute_sample_weight('balanced', y))
        return super().fit(X, y, **kwargs)
```

---

## 8. Results

### Held-Out Test Set (1,913 KOIs — never seen during training)

| Metric | Value |
|---|---|
| **Accuracy** | **81.3%** |
| **Macro F1** | **78.8%** |
| Weighted F1 | 81.5% |
| Macro Recall | 79.6% |
| Macro Precision | 78.3% |
| **5-fold CV Macro F1** | **78.6% ± 0.7%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 🟡 CANDIDATE | 0.618 | 0.669 | **0.642** | 396 |
| 🟢 CONFIRMED | 0.845 | 0.887 | **0.866** | 549 |
| 🔴 FALSE POSITIVE | 0.885 | 0.831 | **0.857** | 968 |

### Confusion Matrix

```
                    Predicted
                  CANDIDATE   CONFIRMED   FALSE POS
Actual CANDIDATE  [  265  ]   [  48   ]   [  83   ]
Actual CONFIRMED  [  41   ]   [  487  ]   [  21   ]
Actual FALSE POS  [  123  ]   [  41   ]   [  804  ]
```

**Why CANDIDATE has the lowest F1 (64.2%) — and why that is correct:**  
CANDIDATE literally means *"we don't know yet."* These KOIs passed automated screening but haven't been confirmed or ruled out by follow-up observations. They sit in the genuine scientific grey zone between the other two classes. The model cannot know what ground-based follow-up will reveal — no ML model can. A 64.2% F1 on the most ambiguous class is not a failure; it correctly reflects real scientific uncertainty.

### Individual Model vs Ensemble

| Model | Accuracy | Macro F1 |
|---|---|---|
| XGBoost alone | ~79.2% | ~77.9% |
| HistGradientBoosting alone | ~78.5% | ~77.3% |
| Random Forest alone | ~77.1% | ~74.8% |
| **Ensemble (all three)** | **81.3%** | **78.8%** |

The ensemble outperforms every individual model because the three algorithms make *different errors* on different subsets of KOIs.

### Improvement Over Baseline

| Model | Macro F1 | Accuracy |
|---|---|---|
| Baseline (25-tree Random Forest, no feature engineering) | 76.8% | 79.2% |
| **Celesta Ensemble (v2)** | **78.8%** | **81.3%** |
| Improvement | **+2.0%** | **+2.1%** |

---

## 9. SHAP Explainability

SHAP (SHapley Additive exPlanations) answers: *for a given prediction, how much did each feature contribute?*

Rooted in cooperative game theory, SHAP distributes the model's output fairly among all features — like splitting a team's prize weighted by each member's marginal contribution. It gives both:

- **Global importance** — which features matter most on average
- **Local explanations** — why the model made one specific prediction

### Top 15 Features by SHAP Importance

| Rank | Feature | SHAP Score | Type |
|---|---|---|---|
| 1 | `koi_max_mult_ev` | 0.548 | Raw — multi-event statistic |
| 2 | `koi_prad` | 0.339 | Raw — planet radius |
| 3 | `koi_count` | 0.315 | Raw — KOI detection count |
| 4 | `single_multi_ratio` | 0.264 | **Engineered ✦** |
| 5 | `duration_period_ratio` | 0.203 | **Engineered ✦** |
| 6 | `koi_model_snr` | 0.182 | Raw — transit SNR |
| 7 | `koi_smet` | 0.166 | Raw — stellar metallicity |
| 8 | `koi_bin_oedp_sig` | 0.156 | Raw — odd-even significance |
| 9 | `koi_ror` | 0.148 | Raw — radius ratio |
| 10 | `koi_max_sngle_ev` | 0.136 | Raw — single-event statistic |

**2 of the top 5 features are engineered** — proving the feature engineering added real predictive value, not noise.

### What the Model Actually Learned

**`koi_max_mult_ev` — The Most Important Signal**

A real planet transits like clockwork — every orbit, the star dims by the exact same amount. The multi-event statistic accumulates this repeating signal proportionally to the square root of transit count. A cosmic ray glitch, an eclipsing binary, or an instrument artefact doesn't repeat consistently. Its multi-event statistic stays low even if one individual transit looked spectacular.

**`single_multi_ratio` — The False Positive Fingerprint (Engineered, Rank #4)**

When a signal fires once spectacularly but doesn't recur, the single-event statistic is high and the multi-event statistic barely grows. The ratio catches exactly this asymmetry. For real planets, both statistics scale together — the ratio stays in a predictable band. For one-off events, the ratio spikes. This single ratio ranked #4 out of 39 features.

**`koi_smet` — A Surprise Finding (Rank #7)**

Stellar metallicity ranked seventh. This is astrophysically meaningful: metal-rich stars are significantly more likely to host planets because planet formation requires heavy elements in the protoplanetary disk. The model discovered this real statistical correlation from data alone, without being programmed to look for it.

**Why Engineered Features Matter**

`duration_period_ratio` (rank #5) encodes the transit chord length, which through Kepler's Third Law ties directly to the mean stellar density. This density provides an independent physical check on the transit geometry — inconsistencies between the photometric density and the spectroscopic stellar parameters flag false positives in exactly the way a human astrophysicist would notice.

### Plain-English Explanation for a General Audience

Imagine Kepler as a security camera watching a field of 150,000 light bulbs. Every time a bulb flickers — dims for a moment — the camera logs it. Our job is to sort those flickers into three piles:

- 🟢 *"A small object passed in front of the bulb on a regular schedule — that's a planet."*
- 🔴 *"The bulb flickered in a way no planet could produce — that's something else."*
- 🟡 *"The flicker looks somewhat planet-like but we can't be sure without closer inspection."*

The key rule the model learned: real planet crossings happen like clockwork. If the bulb dims by the same amount every 2.2 days (or 365 days, or whatever its period is), exactly and repeatedly, that's a planet. If it dims once dramatically and never repeats — or dims differently on alternating crossings — that's almost certainly not a planet. The `single_multi_ratio` feature captures this intuition mathematically and became the fourth most important feature in the model.

---

## 10. Live Web Application

Celesta isn't just a notebook — it deploys as a full web application with:

### 🌐 Interactive Prediction
Fill in any subset of the 27 Kepler features and get an instant classification:
- Predicted disposition (CONFIRMED / CANDIDATE / FALSE POSITIVE)
- Confidence score (0–100%)
- Full probability distribution across all three classes
- List of any missing fields (the ensemble handles them gracefully via NaN)

### 🔭 Explore the Universe
Search any star, exoplanet, nebula, or galaxy and get:
- Wikipedia article extract
- High-resolution NASA Images API photography
- Direct Wikipedia link

### 📊 Performance Dashboard
- Live confusion matrix (3×3)
- Per-class precision, recall, F1
- SHAP feature importance bar chart
- Cross-validation results

### 🪐 3D Hero Visualisation
Interactive Three.js planet rendered in the browser — orbit controls, atmospheric glow, real planetary texture.

### REST API
```bash
# Classify a KOI from raw measurements
curl -X POST /api/predict \
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
    "CANDIDATE":      0.0512,
    "CONFIRMED":      0.8731,
    "FALSE POSITIVE": 0.0757
  },
  "missing_fields": ["koi_time0bk", "koi_srho", "..."]
}
```

---

## 11. Run It Yourself

### Option A — Google Colab (Recommended, zero setup)

Open `notebooks/Celesta_Submission.ipynb` in Google Colab. Click **Runtime → Run all**. The notebook downloads the NASA dataset automatically and runs the full pipeline end-to-end in ~10–15 minutes on a free T4 instance.

### Option B — Run the Web App Locally

**Requirements:** Python 3.9–3.11

```bash
# 1. Clone the repository
git clone https://github.com/vvsrinath/Celesta-India-High-School-Exoplanet-Data-Challenge.git
cd celesta

# 2. Install runtime dependencies
pip install -r requirements.txt

# 3. Start the server  (model.joblib is pre-trained and included)
python3 app.py
```

Open **http://localhost:5000**

### Option C — Retrain the Model from Scratch

```bash
# Install training dependencies (adds shap, imbalanced-learn)
pip install -r requirements-train.txt

# Train — downloads data, engineers features, trains ensemble, writes model.joblib
# Takes 3–5 minutes
python3 train_model.py

# Start the server
python3 app.py
```

### Option D — Docker (trains model inside the build)

```bash
docker build -t celesta .
docker run -p 5000:5000 celesta
```

The multi-stage Dockerfile trains the model in stage 1 and copies only the serialised artefacts into the lean runtime image.

---

## 12. Project Structure

```
celesta/
├── app.py                        # Flask server — routes, prediction, Explore API
├── train_model.py                # Training script — data → features → ensemble → model.joblib
├── model_utils.py                # BalancedXGBClassifier — shared by train and app
├── model.joblib                  # Pre-trained ensemble (~47 MB, gitignored)
├── requirements.txt              # Runtime dependencies
├── requirements-train.txt        # Additional training dependencies (shap, imbalanced-learn)
├── Dockerfile                    # Multi-stage: train in stage 1, serve in stage 2
│
├── data/
│   ├── koi_stripped.csv          # Cleaned training CSV (1.86 MB)
│   └── stats.json                # Pre-computed metrics served to the frontend
│
├── static/
│   ├── css/style.css             # Dark space theme, fully responsive
│   ├── js/app.js                 # Three.js hero, Explore section, prediction form
│   ├── model.json                # Feature definitions for the prediction form UI
│   └── images/                  # Planet textures, star photos, favicon
│
├── templates/
│   └── index.html               # Jinja2 template — rendered by Flask on GET /
│
├── notebooks/
│   ├── Celesta_Submission.ipynb  # ← Hackathon submission notebook (run this)
│   └── Celesta_Colab.ipynb       # Extended notebook with additional analysis
│
├── docs/                         # Static GitHub Pages demo (no Python needed)
└── wiki/                         # Architecture, API reference, model card
```

---

## 13. API Reference

All endpoints return JSON. Every response carries `Cache-Control: no-store` headers.

### `GET /api/stats`

Returns the full model metrics object from `data/stats.json`.

**Response:**
```json
{
  "accuracy": 0.8134,
  "macro_f1": 0.7884,
  "cv_macro_f1_mean": 0.7865,
  "cv_macro_f1_std": 0.0071,
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
  "per_class": {
    "CANDIDATE":      {"precision": 0.618, "recall": 0.669, "f1": 0.642},
    "CONFIRMED":      {"precision": 0.845, "recall": 0.887, "f1": 0.866},
    "FALSE POSITIVE": {"precision": 0.885, "recall": 0.831, "f1": 0.857}
  },
  "feature_importance": [["koi_max_mult_ev", 0.548], ...]
}
```

### `POST /api/predict`

Send any subset of the 27 raw features as JSON. Missing values are handled gracefully (NaN).

### `GET /api/explore/search?q=orion`

Searches Wikipedia for astronomy topics — returns up to 8 `{title, description}` pairs.

### `GET /api/explore/details?name=Orion+Nebula`

Returns full object details: Wikipedia extract, description, NASA Images API photo, Wikipedia link.

---

## 14. Tech Stack

| Layer | Technology |
|---|---|
| **ML** | XGBoost 2.x · scikit-learn 1.9 (HistGradientBoosting, RandomForest, Pipeline) |
| **Explainability** | SHAP (TreeExplainer on XGBoost sub-model) |
| **Backend** | Python 3.11 · Flask 3.1 · Gunicorn |
| **Frontend** | Three.js · Vanilla JS · CSS custom properties |
| **Data** | pandas · NumPy · NASA Exoplanet Archive |
| **Serialisation** | joblib |
| **Deployment** | Docker (multi-stage) · Gunicorn WSGI |

---

## 15. Credits

**Developed by Srinath Vatchavari Venkateshan**

**Source Code:** [GitHub](https://github.com/vvsrinath/Celesta-India-High-School-Exoplanet-Data-Challenge)  
**Live Demo:** [GitHub Pages](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)  
**Dataset:** NASA Exoplanet Archive — Kepler Objects of Interest Cumulative Table (public domain)  
**SHAP:** Lundberg & Lee, 2017 — [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)  
**XGBoost:** Chen & Guestrin, 2016 — [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)  
**Challenge:** India High School Exoplanet Data Challenge

---

<div align="center">

*Made with curiosity, code, and a lot of starlight.*

**Srinath Vatchavari Venkateshan**

[GitHub](https://github.com/vvsrinath) | [LinkedIn](https://www.linkedin.com/in/srinath-v-a26b372b7/) | [Email](mailto:vvsrinath0@gmail.com)

</div>
