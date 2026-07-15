# Contributing

This page covers every way to run, develop, and extend Celesta.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Running locally](#2-running-locally)
3. [Running with Docker](#3-running-with-docker)
4. [GitHub Pages demo](#4-github-pages-demo)
5. [Using the Colab notebook](#5-using-the-colab-notebook)
6. [Project structure for contributors](#6-project-structure-for-contributors)
7. [Adding a new raw feature](#7-adding-a-new-raw-feature)
8. [Adding an engineered feature](#8-adding-an-engineered-feature)
9. [Retraining the model](#9-retraining-the-model)
10. [Code style](#10-code-style)

---

## 1. Prerequisites

- Python 3.10 or newer
- pip

No virtual environment is required, but you can use one if you prefer.

---

## 2. Running locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/celesta.git
cd celesta

# Install dependencies
pip install -r requirements.txt

# Train the model (takes ~3 minutes on a modern CPU)
python3 train_model.py

# Start the Flask server
python3 app.py
```

Open `http://localhost:5000` in your browser.

The training step is only needed once. After that, `model.joblib` is cached on disk and loaded at app startup. If you change any features or the dataset, you must retrain.

### What train_model.py does

1. Loads `data/koi_stripped.csv`.
2. Applies the 8 engineered features via `add_features()`.
3. Trains a soft-voting ensemble (XGBoost + HistGradientBoosting + RandomForest) with balanced class weights.
4. Evaluates on a held-out 20% test split and runs 5-fold cross-validation.
5. Computes SHAP values on the XGBoost sub-model.
6. Saves `model.joblib` (~49 MB, gitignored) and `data/stats.json`.
7. Writes `data/koi_stripped.csv` (the raw-features-only version used by the notebooks).

---

## 3. Running with Docker

The `Dockerfile` is a multi-stage build so the final image does not contain the large training packages.

```bash
cd celesta
docker build -t celesta .
docker run -p 5000:5000 celesta
```

Open `http://localhost:5000`.

**How the build works:**

- **Stage 1 (`trainer`):** Installs all ML packages, runs `train_model.py`, produces `model.joblib` and `data/stats.json`.
- **Stage 2 (`runner`):** Installs only Flask + runtime dependencies, copies `model.joblib` and `stats.json` from stage 1, runs `app.py`.

The `.dockerignore` file excludes `.git/`, `docs/`, `*.ipynb`, `wiki/`, `__pycache__/`, and `model.joblib` from the build context (the model is trained inside Docker, not copied in).

---

## 4. GitHub Pages demo

The `docs/` directory contains a static HTML demo. Enabling it:

1. Push the repo to GitHub.
2. Go to **Settings → Pages**.
3. Set source to **branch `main`**, folder **`/docs`**.
4. Click **Save**. GitHub Pages will publish at `https://<username>.github.io/celesta/`.

The static demo uses hardcoded example predictions and the stats from `data/stats.json` — it does not run the live model (that requires a running Flask server).

Cache-busting meta tags and `?v=2` query strings on CSS/JS links are already in `docs/index.html` to prevent browsers from serving stale versions after updates.

---

## 5. Using the Colab notebook

`Celesta_Colab.ipynb` is fully self-contained:

1. Open it in [Google Colab](https://colab.research.google.com/).
2. Click **Runtime → Run all**.
3. The notebook downloads the KOI table directly from the NASA Exoplanet Archive API, trains the full pipeline, evaluates it, and generates SHAP plots — no file uploads needed.

`Celesta_Hackathon_Notebook.ipynb` is the local version:

1. Make sure `data/koi_stripped.csv` exists (it is tracked in git, so it should already be there after cloning).
2. Run all cells.

---

## 6. Project structure for contributors

```
app.py              Flask server. Do not add ML logic here.
train_model.py      Training script. All ML logic lives here.
model_utils.py      BalancedXGBClassifier wrapper. Import it in both scripts.
data/
  koi_stripped.csv  Source dataset. Do not modify manually.
  stats.json        Written by train_model.py. Do not modify manually.
templates/
  index.html        Jinja2 template. Edit UI here.
static/
  css/style.css     Stylesheet.
  js/main.js        Frontend JavaScript.
  images/           Static assets.
docs/
  index.html        GitHub Pages static demo. Keep in sync with templates/index.html.
```

---

## 7. Adding a new raw feature

1. **Check the KOI table** at the [NASA Exoplanet Archive column descriptions](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html) to confirm the column exists and is not a leakage risk (no disposition flags, no follow-up results).

2. **Add it to `RAW_FEATURES`** in `train_model.py`:
   ```python
   RAW_FEATURES = [
       "koi_period", ...,
       "your_new_column",   # add here
   ]
   ```

3. **Add it to the `input_fields` block** in `templates/index.html` so users can enter a value in the web UI.

4. **Retrain** (`python3 train_model.py`) — this updates `model.joblib` and `stats.json`.

5. **Test** with a curl request to `POST /api/predict` including the new field.

---

## 8. Adding an engineered feature

1. **Add the computation** to the `add_features()` function in `train_model.py`:
   ```python
   def add_features(df):
       ...
       d["your_feature"] = d["raw_col_a"] / (d["raw_col_b"] + eps)
       return d
   ```

2. **Add the same computation** to the `_add_features()` function in `app.py` (note the leading underscore). The two functions must stay identical — the app applies this at inference time.

3. **Add the name** to the `ENGINEERED` list in `train_model.py`:
   ```python
   ENGINEERED = [
       "single_multi_ratio", ...,
       "your_feature",
   ]
   ```

4. **Retrain** and verify it appears in the SHAP importance output.

> **Important:** Do not send engineered features to `/api/predict`. The server computes them from the raw inputs. The `engineered` key in `model.joblib` tells the server which features to compute automatically.

---

## 9. Retraining the model

Any time you change features, hyperparameters, or the dataset:

```bash
python3 train_model.py
```

Then restart the Flask server so it loads the new `model.joblib`:

```bash
python3 app.py
```

The server loads the model once at startup and caches it in `_model_cache`. You must restart the process — a hot-reload is not enough.

---

## 10. Code style

- Python files use 4-space indentation and follow PEP 8 loosely.
- Comments explain *why*, not *what* — the code itself explains what.
- No AI-style banner comments (`# ── Section ───────────`). Use plain headings.
- No encyclopedic docstrings for trivial functions.
- `BalancedXGBClassifier` must always live in `model_utils.py` — moving it breaks joblib unpickling in `app.py`. See the [Model Card](Model-Card) for the reason.
