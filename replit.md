# Celesta — Kepler Exoplanet Classifier

A leakage-free machine learning web app that classifies Kepler Objects of Interest (KOIs) as **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** using only raw telescope measurements — no NASA vetting outputs.

**Stack:** Python 3.11 · Flask 3.1 · XGBoost 2.x · scikit-learn 1.9 · Three.js

## How to run

```bash
cd celesta
python3 app.py
```

Open http://localhost:5000

## Project layout

```
celesta/
├── app.py               # Flask server — routes and prediction API
├── train_model.py       # Training script (run once to regenerate model.joblib)
├── model_utils.py       # BalancedXGBClassifier — shared by train + app
├── model.joblib         # Pre-trained ensemble (~47 MB, generated binary)
├── requirements.txt     # Runtime dependencies
├── requirements-train.txt  # Extra deps for training (shap, imbalanced-learn)
├── data/
│   ├── koi_stripped.csv     # Cleaned training CSV (1.86 MB)
│   └── stats.json           # Pre-computed metrics served to frontend
├── static/
│   ├── css/style.css        # Dark space theme, responsive
│   ├── js/app.js            # Three.js hero, Explore section, prediction form
│   ├── model.json           # Feature definitions for the prediction form
│   └── images/              # Textures, favicon, developer photo
├── templates/
│   └── index.html           # Jinja2 template rendered by Flask
├── notebooks/
│   ├── Celesta_Colab.ipynb          # Self-contained Google Colab notebook
│   └── Celesta_Hackathon_Notebook.ipynb  # Local training notebook
├── docs/                    # Static GitHub Pages demo (no Flask required)
└── wiki/                    # Architecture, API reference, model card docs
```

## Source size

Excluding the generated `model.joblib` binary: **~2.9 MB**

## Regenerating the model

The pre-trained `model.joblib` is already included. To retrain from scratch:

```bash
cd celesta
pip install -r requirements-train.txt   # adds shap, imbalanced-learn
python3 train_model.py                  # ~3-5 minutes, rewrites model.joblib + data/stats.json
```

## API

- `GET /api/stats` — model metrics (accuracy, F1, confusion matrix, SHAP importance)
- `POST /api/predict` — classify a KOI; accepts any subset of 27 raw features as JSON
- `GET /api/explore/search?q=orion` — Wikipedia astronomy search
- `GET /api/explore/details?name=Orion+Nebula` — Wikipedia + NASA image details

## Optional: Google Custom Search

Set `GOOGLE_CSE_CX` and `GOOGLE_CSE_API_KEY` environment variables to enable richer images in the Explore section.

## User preferences

- Notebooks live in `celesta/notebooks/`
- Keep project source under 5 MB (model.joblib is a generated binary and excluded from this target)
