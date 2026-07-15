# Celesta — Kepler Exoplanet Classifier

## Project Overview

A Flask web app that classifies NASA Kepler exoplanet candidates (CONFIRMED / CANDIDATE / FALSE POSITIVE) using a leakage-free Random Forest trained on raw transit and stellar signals. Features a Three.js 3D animated hero, an Explore the Universe section (Wikipedia + NASA Images API), a live prediction form, and a performance dashboard.

- **Entry point:** `celesta/app.py`
- **Run command:** `cd celesta && python3 app.py`
- **Port:** 5000
- **Pre-trained model:** `celesta/model.joblib` — included, no retraining needed

## Tech Stack

- Python 3.11, Flask 3.1
- scikit-learn 1.9, pandas 3, numpy 2, joblib
- Three.js r134 (CDN)
- Wikipedia REST API, NASA Images API (Explore section)

## Key Files

```
celesta/
├── app.py              # Flask backend — routes, prediction, Explore API
├── train_model.py      # Model training (optional — model already included)
├── model.joblib        # Pre-trained pipeline + label encoder
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build (gunicorn, port 5000)
├── Procfile            # Railway / Heroku
├── render.yaml         # Render one-click deploy
├── data/stats.json     # Pre-computed metrics served to frontend
├── static/             # CSS, JS, textures, favicon
├── templates/          # Jinja2 HTML template
└── docs/               # Static GitHub Pages snapshot (deploy docs/ folder)
```

## Environment Variables (all optional)

| Variable | Purpose |
|---|---|
| `GOOGLE_CSE_CX` | Google Custom Search Engine ID (Explore section images) |
| `GOOGLE_CSE_API_KEY` | Google Custom Search API key |

The app runs fully without these — image search silently skips if unset.

## Deployment

- **GitHub Pages:** push `docs/` branch → Settings → Pages → `/docs` folder
- **Render:** `render.yaml` auto-configures → one-click deploy
- **Docker:** `docker build -t celesta . && docker run -p 5000:5000 celesta`
- **Replit:** workflow `Start application` runs `cd celesta && python3 app.py`

## Project Size Notes

`data/koi_stripped.csv` (~1.9 MB) is gitignored — it's training-only data.  
Tracked repo size ≈ 2.8 MB (under the 5 MB target).

## User Preferences

- Keep project size under 5 MB (training CSV is gitignored)
- GitHub Pages static demo lives in `docs/` — do not break this folder
- No Google Custom Search keys — app must work without them
- Docker support required (gunicorn, port 5000)
