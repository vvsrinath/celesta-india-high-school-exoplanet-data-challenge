# Celesta — Wiki

**Celesta** is a machine-learning web application that classifies NASA Kepler Objects of Interest (KOIs) as **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** using only raw transit and stellar measurements. No derived disposition flags are used as inputs, so the model cannot leak the answer to itself.

Built for the **India High School Exoplanet Data Challenge** by **Srinath V Venkateshan**.

---

## Contents

| Page | What it covers |
|---|---|
| [Home](Home) | This page — project overview and navigation |
| [System Architecture](System-Architecture) | Full data flow diagrams: training pipeline, inference pipeline, Explore API chain, Docker build, frontend, cache strategy |
| [API Reference](API-Reference) | Every Flask route: URL, method, parameters, response shape, examples |
| [Model Card](Model-Card) | Features, pipeline architecture, metrics, SHAP importance, known limitations |
| [Contributing](Contributing) | How to run locally, train the model, add features, run tests |
| [Changelog](Changelog) | Version history |

---

## Problem Statement

The NASA Exoplanet Archive Kepler Objects of Interest table lists ~9 500 candidate planetary signals found in Kepler photometry. Each KOI is labelled:

- **CONFIRMED** — follow-up observations proved it is a real planet.
- **FALSE POSITIVE** — follow-up observations proved it is NOT a planet (background eclipsing binary, instrumental artefact, etc.).
- **CANDIDATE** — follow-up observations have not yet resolved it either way.

The challenge is to predict the label from the raw transit and stellar parameters alone — the same information a telescope pipeline sees before any human review.

---

## Quick Start

```bash
git clone https://github.com/<your-username>/celesta.git
cd celesta
pip install -r requirements.txt
python3 train_model.py   # trains the model, saves model.joblib
python3 app.py           # starts Flask on http://localhost:5000
```

See [Contributing](Contributing) for Docker and GitHub Pages instructions.

---

## Repository Layout

```
celesta/
├── app.py                  # Flask server + API routes
├── train_model.py          # Training script — run once before starting the app
├── model_utils.py          # BalancedXGBClassifier (pickle-compatible wrapper)
├── requirements.txt
├── Dockerfile              # Multi-stage build: train → slim runtime
├── .dockerignore
├── .gitignore
├── data/
│   ├── koi_stripped.csv    # Cleaned Kepler dataset (1.86 MB, tracked in git)
│   └── stats.json          # Model metrics written after training
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/             # Textures, favicon, developer photo
├── templates/
│   └── index.html          # Jinja2 template served by Flask
├── docs/
│   └── index.html          # Static GitHub Pages demo
├── Celesta_Colab.ipynb     # Self-contained Google Colab notebook
├── Celesta_Hackathon_Notebook.ipynb  # Local notebook (requires koi_stripped.csv)
└── wiki/                   # This GitHub Wiki (push to <repo>.wiki.git)
```

---

## Author

**Srinath V Venkateshan**
[LinkedIn](https://www.linkedin.com/in/srinath-v-a26b372b7/) · [vvsrinath0@gmail.com](mailto:vvsrinath0@gmail.com) · [GitHub](https://github.com/vvsrinath)
