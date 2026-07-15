# Model Card

**Model name:** Celesta Exoplanet Classifier v1  
**Task:** Multi-class classification — CANDIDATE / CONFIRMED / FALSE POSITIVE  
**Dataset:** NASA Exoplanet Archive — Kepler Objects of Interest (KOI) Cumulative Table  
**Author:** Srinath V Venkateshan  

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Features](#2-features)
3. [Feature Engineering](#3-feature-engineering)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Training Setup](#5-training-setup)
6. [Metrics](#6-metrics)
7. [SHAP Feature Importance](#7-shap-feature-importance)
8. [Known Limitations](#8-known-limitations)
9. [Ethical Considerations](#9-ethical-considerations)

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — KOI Cumulative Table |
| Local file | `data/koi_stripped.csv` |
| Total rows | 9 564 |
| Train split | 7 651 (80%) |
| Test split | 1 913 (20%) |
| Split strategy | Stratified by class label |

**Class distribution (full dataset):**

| Class | Count | Fraction |
|---|---|---|
| CANDIDATE | 1 978 | 20.7% |
| CONFIRMED | 2 747 | 28.7% |
| FALSE POSITIVE | 4 839 | 50.6% |

The dataset is imbalanced. FALSE POSITIVE KOIs outnumber CANDIDATE KOIs 2.4:1. All three models in the ensemble use class-weight correction to prevent the majority class from dominating training.

**No data leakage:** The following columns are explicitly excluded because they encode the target or are derived from follow-up observations not available at transit-detection time:

- `koi_disposition` (the target itself)
- `koi_pdisposition` (pipeline disposition — derived from the target)
- Any `koi_fpflag_*` columns (false-positive flag fields)
- `kepoi_name`, `kepler_name`, `koi_comment` (identifiers and free text)
- `koi_sage`, `koi_model_dof`, `koi_model_chisq` (100% missing in this dataset)

---

## 2. Features

### Raw features (27)

| Feature | Description | Units |
|---|---|---|
| `koi_period` | Orbital period | days |
| `koi_time0bk` | Time of first transit | days (BJD − 2 454 833) |
| `koi_impact` | Impact parameter — 0 = central transit, >1 = no transit | dimensionless |
| `koi_duration` | Transit duration | hours |
| `koi_depth` | Fractional flux decrease at transit midpoint | ppm |
| `koi_ror` | Planet-to-star radius ratio (Rp/R★) | dimensionless |
| `koi_srho` | Fitted stellar density from transit shape | g/cm³ |
| `koi_prad` | Planet radius derived from `koi_ror` and `koi_srad` | Earth radii |
| `koi_sma` | Orbital semi-major axis | AU |
| `koi_incl` | Orbital inclination | degrees |
| `koi_teq` | Planetary equilibrium temperature | K |
| `koi_insol` | Insolation flux (irradiance relative to Earth) | Earth flux |
| `koi_dor` | Semi-major axis normalised by stellar radius (a/R★) | dimensionless |
| `koi_max_sngle_ev` | Maximum single-event statistic | σ |
| `koi_max_mult_ev` | Maximum multiple-event statistic (MES) | σ |
| `koi_model_snr` | Transit model signal-to-noise ratio | dimensionless |
| `koi_count` | Number of KOIs associated with the same star | integer |
| `koi_num_transits` | Number of observed transits contributing to detection | integer |
| `koi_bin_oedp_sig` | Significance of odd-even depth difference | σ |
| `koi_steff` | Stellar effective temperature | K |
| `koi_slogg` | Stellar surface gravity | log₁₀(cm/s²) |
| `koi_smet` | Stellar metallicity ([Fe/H]) | dex |
| `koi_srad` | Stellar radius | solar radii |
| `koi_smass` | Stellar mass | solar masses |
| `ra` | Right ascension of the host star | degrees |
| `dec` | Declination of the host star | degrees |
| `koi_kepmag` | Kepler-band magnitude of the host star | mag |

---

## 3. Feature Engineering

Eight features are derived from the raw inputs. They are computed identically in `train_model.py` (at training time) and `app.py` (at inference time).

| Feature | Formula | Physical meaning |
|---|---|---|
| `single_multi_ratio` | `koi_max_sngle_ev / (koi_max_mult_ev + ε)` | High ratio → single isolated event → likely not a repeating planet |
| `duration_period_ratio` | `koi_duration / (koi_period × 24 + ε)` | Encodes transit chord length; ties into stellar density via Kepler III |
| `log_period` | `log1p(koi_period)` | Period is right-skewed; log transform makes the distribution more Gaussian |
| `log_depth` | `log1p(koi_depth)` | Transit depth spans orders of magnitude; log reduces dynamic range |
| `log_snr` | `log1p(koi_model_snr)` | Same motivation as log_depth |
| `stellar_density_proxy` | `koi_smass / (koi_srad³ + ε)` | Proportional to mean stellar density; independent check on transit geometry |
| `impact_ror_ratio` | `koi_impact / (koi_ror + ε)` | Values near 1 → grazing transit → common false-positive configuration |
| `expected_duration_ratio` | `koi_duration / (T_expected + ε)` | Compares observed duration to Keplerian prediction; large deviations suggest eccentricity or blended binaries |

Where `T_expected = (P × 24 / π) × (Rp/R★) / (a/R★) × √(1 − b²)` and ε = 1 × 10⁻⁹.

---

## 4. Pipeline Architecture

The model is a **soft-voting ensemble** of three base learners, stored as a scikit-learn `VotingClassifier`. It is saved to `model.joblib` using joblib and loaded once at app startup.

```
VotingClassifier (voting="soft")
├── xgb  →  BalancedXGBClassifier
│            XGBoost, 500 trees, depth 6, lr 0.04
│            sample_weight="balanced" (computed at fit time)
│            Handles NaN natively via hist tree method
│
├── hgb  →  HistGradientBoostingClassifier
│            500 iterations, depth 8, lr 0.04, min_samples_leaf 20
│            class_weight="balanced"
│            Handles NaN natively
│
└── rf   →  Pipeline
             ├── SimpleImputer (strategy="median")
             └── RandomForestClassifier
                  400 trees, depth 16, max_features="sqrt"
                  class_weight="balanced_subsample"
```

**`BalancedXGBClassifier`** (`model_utils.py`) is a thin wrapper around `XGBClassifier` that auto-applies `compute_sample_weight("balanced", y)` at fit time, working around scikit-learn 1.9's metadata-routing requirement inside `VotingClassifier`.

**Why this ensemble?** The three learners make different assumptions: XGBoost uses second-order gradients and is strong on tabular data; HistGradientBoosting is a fast, regularised alternative that adds diversity; Random Forest adds high-variance, low-bias estimates through bagging. Averaging their soft probabilities reduces variance without increasing bias significantly.

**joblib pickle file layout:**

```python
{
    "pipeline":      VotingClassifier,   # the full ensemble
    "label_encoder": LabelEncoder,       # maps int predictions back to class names
    "features":      list[str],          # 35 feature names in order
    "engineered":    list[str],          # 8 engineered feature names
}
```

---

## 5. Training Setup

| Setting | Value |
|---|---|
| Random seed | 42 |
| Train/test split | 80/20, stratified |
| Cross-validation | 5-fold stratified, scoring=`f1_macro` |
| Hyperparameter tuning | Manual (not Optuna) |
| Explainability | SHAP `TreeExplainer` on XGBoost sub-model (500-sample background) |

Run training:

```bash
python3 train_model.py
```

Output: `model.joblib` (~49 MB, gitignored), `data/stats.json`, `data/koi_stripped.csv`.

---

## 6. Metrics

Results on the held-out 20% test split (1 913 samples):

| Metric | Value |
|---|---|
| Accuracy | **81.34%** |
| Macro F1 | **78.84%** |
| Weighted F1 | **81.52%** |
| Macro Recall | **79.56%** |
| Macro Precision | **78.29%** |
| 5-fold CV Macro F1 | **78.65% ± 0.71%** |

**Per-class metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| CANDIDATE | 0.618 | 0.669 | **0.642** | 396 |
| CONFIRMED | 0.845 | 0.887 | **0.866** | 549 |
| FALSE POSITIVE | 0.885 | 0.831 | **0.857** | 968 |

**Confusion matrix** (rows = true, columns = predicted):

| | CANDIDATE | CONFIRMED | FALSE POSITIVE |
|---|---|---|---|
| **CANDIDATE** | 265 | 48 | 83 |
| **CONFIRMED** | 41 | 487 | 21 |
| **FALSE POSITIVE** | 123 | 41 | 804 |

CANDIDATE is the hardest class because it represents genuine scientific uncertainty — NASA uses the label to mean "not yet resolved by follow-up". The model's 64.2% F1 on CANDIDATEs correctly reflects this ambiguity.

---

## 7. SHAP Feature Importance

Computed via `shap.TreeExplainer` on the XGBoost sub-model using a 500-sample background set. Values are mean |SHAP| averaged across all classes and samples.

| Rank | Feature | Mean |SHAP| | Type |
|---|---|---|---|
| 1 | `koi_max_mult_ev` | 0.548 | Raw |
| 2 | `koi_prad` | 0.339 | Raw |
| 3 | `koi_count` | 0.315 | Raw |
| 4 | `single_multi_ratio` | 0.264 | **Engineered** |
| 5 | `duration_period_ratio` | 0.203 | **Engineered** |
| 6 | `koi_model_snr` | 0.182 | Raw |
| 7 | `koi_smet` | 0.166 | Raw |
| 8 | `koi_bin_oedp_sig` | 0.156 | Raw |
| 9 | `koi_ror` | 0.148 | Raw |
| 10 | `koi_max_sngle_ev` | 0.136 | Raw |
| 11 | `koi_impact` | 0.126 | Raw |
| 12 | `ra` | 0.119 | Raw |
| 13 | `koi_duration` | 0.111 | Raw |
| 14 | `koi_dor` | 0.108 | Raw |
| 15 | `dec` | 0.105 | Raw |

Two of the top 5 features are engineered. `single_multi_ratio` (rank 4) captures whether a signal is a one-off versus a repeating pattern — the clearest physical test for a false positive. `duration_period_ratio` (rank 5) encodes transit geometry in a way that is directly tied to stellar density through Kepler's third law.

---

## 8. Known Limitations

- **CANDIDATE recall is 66.9%.** This is a fundamental limit: CANDIDATE KOIs are those NASA has not yet resolved, so any model trained on the same data cannot know the true answer.
- **Hyperparameters are hand-tuned.** Bayesian optimisation (e.g. Optuna) would likely improve results by 1–2% F1.
- **Feature set is fixed at training time.** If NASA adds new columns to the KOI table, the model must be retrained.
- **No temporal validation.** The train/test split is random, not time-ordered. A model trained only on early Kepler data and tested on later data might perform worse.
- **`koi_srho` is ~40% missing** in the dataset and is imputed to the median by the RandomForest pipeline stage. Rows with many missing fields will be predicted with higher uncertainty.

---

## 9. Ethical Considerations

- The model is a **decision-support tool**, not a replacement for follow-up observations.
- Predictions marked CONFIRMED should not be published as confirmed exoplanet discoveries — they require independent telescope confirmation.
- The training data reflects observational biases in the Kepler mission (field of view, magnitude limit, cadence). The model inherits these biases.
