# Changelog

All notable changes to Celesta are recorded here.

Format: **[version] — date** followed by added / changed / fixed / removed sections.

---

## [1.0.0] — 2026-07-12

Initial public release for the India High School Exoplanet Data Challenge.

### Added

- **Soft-voting ensemble** — XGBoost + HistGradientBoosting + RandomForest with balanced class weights. Achieves 81.3% accuracy and 78.8% macro F1 on the held-out test split.
- **8 engineered features** — `single_multi_ratio`, `duration_period_ratio`, `log_period`, `log_depth`, `log_snr`, `stellar_density_proxy`, `impact_ror_ratio`, `expected_duration_ratio`. Two of these appear in the top-5 SHAP importance ranking.
- **`BalancedXGBClassifier`** (`model_utils.py`) — thin wrapper around XGBClassifier that auto-applies balanced sample weights at fit time, working around scikit-learn 1.9 metadata-routing restrictions inside `VotingClassifier`.
- **Flask API** — five endpoints: `GET /`, `GET /api/stats`, `POST /api/predict`, `GET /api/explore/search`, `GET /api/explore/details`. See [API Reference](API-Reference).
- **Explore section** — Wikipedia summary, NASA Images, and optional Google Custom Search image lookup for any astronomy object.
- **No-cache response headers** — `Cache-Control: no-cache, no-store, must-revalidate` on every response; `SEND_FILE_MAX_AGE_DEFAULT = 0` for static files.
- **Multi-stage Dockerfile** — stage 1 trains the model inside Docker; stage 2 is a slim runtime image. Final image does not include the 800 MB of training packages.
- **`.dockerignore`** — excludes `.git/`, `docs/`, `wiki/`, `*.ipynb`, `model.joblib`, `__pycache__/`, and other build-context bloat.
- **GitHub Pages demo** (`docs/index.html`) — static demo with cache-busting meta tags and `?v=2` query strings on CSS/JS includes.
- **`Celesta_Colab.ipynb`** — self-contained Google Colab notebook. Downloads the KOI table directly from the NASA Exoplanet Archive API; no file upload needed.
- **`Celesta_Hackathon_Notebook.ipynb`** — local notebook that uses `data/koi_stripped.csv`. Patched to use `BalancedXGBClassifier` and remove LightGBM fit_params.
- **`README.md`** — full rewrite with problem statement, dataset guide, feature table, engineering rationale, metrics, SHAP table, and all run methods.
- **Wiki** (`wiki/`) — Home, API Reference, Model Card, Contributing, Changelog pages.
- **Inline docstrings** — added to `app.py`, `train_model.py`, and `model_utils.py`.

### Removed

- **LightGBM** from the ensemble — `libgomp.so.1` is missing on the target deployment environment (Replit NixOS). LightGBM is still listed in `requirements.txt` but is not imported anywhere; safe to remove if desired.

---

## Pre-release — 2026-07-11

- Initial model with single XGBoost classifier, no feature engineering.
- Basic Flask UI — form inputs, prediction display.
- `data/koi_stripped.csv` added to git as the canonical training input.
- `model.joblib` added to `.gitignore` (49 MB — too large for git).
