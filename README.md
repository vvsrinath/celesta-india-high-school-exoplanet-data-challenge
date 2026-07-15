<div align="center">

<img src="celesta/static/images/title_banner.png" alt="Celesta — Kepler Exoplanet Classifier" width="100%"/>

# Celesta — Kepler Exoplanet Classifier

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

| Accuracy | Macro F1 | CV F1 | Features | KOIs |
|:-----------:|:-----------:|:--------:|:-----------:|:-------:|
| **81.3 %** | **78.8 %** | **78.6 % +/- 0.7 %** | **39** | **9,564** |

<br>

**Developed by Srinath Vatchavari Venkateshan**

[Source Code](https://github.com/vvsrinath/Celesta-India-High-School-Exoplanet-Data-Challenge) | [Live Demo](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)

</div>

---

## Repository

| Folder | What's inside |
|---|---|
| [`celesta/`](celesta/) | Flask web app — model training, prediction API, frontend |
| [`colab/`](colab/) | Google Colab notebooks — full ML pipeline, EDA, SHAP |
| [`docs/`](docs/) | Static GitHub Pages demo site (no Python needed) |

---

## Quick Start

### Option A — Run the web app locally

```bash
cd celesta
pip install -r requirements.txt
python3 app.py          # -> http://localhost:5000
```

### Option B — Docker (trains model inside the build)

```bash
cd celesta
docker build -t celesta .
docker run -p 5000:5000 celesta
```

### Option C — Open the submission notebook in Colab

See [`colab/README.md`](colab/README.md) for Colab badges and instructions.

---

## Live Demo

The project is deployed on GitHub Pages: [https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)

---

## Project Structure

```
Celestazip/
├── celesta/                  # Main application
│   ├── app.py                # Flask server — routes, prediction, Explore API
│   ├── train_model.py        # Training pipeline — data -> features -> ensemble
│   ├── model_utils.py        # BalancedXGBClassifier
│   ├── model.joblib          # Pre-trained ensemble (gitignored, ~47 MB)
│   ├── requirements.txt      # Runtime dependencies
│   ├── Dockerfile            # Multi-stage Docker build
│   ├── data/                 # Training CSV and stats
│   ├── static/               # CSS, JS, images, textures
│   └── templates/            # Jinja2 HTML templates
├── colab/                    # Google Colab notebooks
├── docs/                     # GitHub Pages static site
└── .github/workflows/        # GitHub Actions CI/CD
```

---

## How It Works

1. **Data**: NASA Kepler Objects of Interest (KOI) Cumulative Table — 9,564 records
2. **Features**: 27 raw transit/stellar measurements + 12 engineered physics features
3. **Model**: Soft-voting ensemble of XGBoost, HistGradientBoosting, RandomForest
4. **Performance**: 81.3% accuracy, 78.8% Macro F1 on held-out test set
5. **Deploy**: Flask web app with prediction API, Wikipedia/NASA Explore section, 3D hero

---

## API

```bash
# Classify a KOI
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"koi_period": 2.204, "koi_prad": 1.5, "koi_model_snr": 45.3}'
```

```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.8731,
  "probabilities": {"CANDIDATE": 0.05, "CONFIRMED": 0.87, "FALSE POSITIVE": 0.08}
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | XGBoost, HistGradientBoosting, RandomForest |
| Explainability | SHAP (TreeExplainer) |
| Backend | Python 3.11, Flask, Gunicorn |
| Frontend | Three.js, Vanilla JS, CSS |
| Data | NASA Exoplanet Archive, pandas, NumPy |
| Deployment | Docker, GitHub Pages |

---

## GitHub Pages Deployment

The `docs/` folder is deployed automatically to GitHub Pages via GitHub Actions.

- Push changes to `docs/` on the `main` branch to trigger a deploy
- Or trigger manually from the Actions tab
- Live at: [https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/](https://vvsrinath.github.io/Celesta-India-High-School-Exoplanet-Data-Challenge/)

---

## License

MIT License — see [`celesta/LICENSE`](celesta/LICENSE)

---

<div align="center">

*Made with curiosity, code, and a lot of starlight.*

**Srinath Vatchavari Venkateshan**

[GitHub](https://github.com/vvsrinath/Celesta-India-High-School-Exoplanet-Data-Challenge) | [LinkedIn](https://www.linkedin.com/in/srinath-v-a26b372b7/) | [Email](mailto:vvsrinath0@gmail.com)

</div>
