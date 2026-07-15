# 🌌 Celesta — Google Colab Notebooks

**Developed by Srinath Vatchavari Venkateshan**

This folder contains all Jupyter / Google Colab notebooks for the **Celesta** exoplanet
classification project, kept separate from the web-app source code so the
repository stays under 10 MB.

---

## Notebooks

| File | Purpose | Open in Colab |
|---|---|---|
| **`Celesta_Submission.ipynb`** | ⭐ Hackathon submission — full pipeline, 13 sections, SHAP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/celesta/blob/main/colab/Celesta_Submission.ipynb) |
| `Celesta_Colab.ipynb` | Extended analysis with extra visualisations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/celesta/blob/main/colab/Celesta_Colab.ipynb) |
| `Celesta_Hackathon_Notebook.ipynb` | Development scratchpad / iteration history | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/celesta/blob/main/colab/Celesta_Hackathon_Notebook.ipynb) |

> Replace `YOUR_USERNAME` with your GitHub username after pushing to GitHub.

---

## How to Run

### Option 1 — Google Colab (recommended, zero setup)

1. Click any **Open in Colab** badge above
2. **Runtime → Run all** (Ctrl+F9)
3. The notebook downloads the NASA dataset automatically — no local files needed
4. Full pipeline runs in ~10–15 minutes on a free T4 GPU

### Option 2 — Jupyter locally

```bash
pip install jupyter xgboost shap scikit-learn pandas numpy matplotlib seaborn
jupyter notebook colab/Celesta_Submission.ipynb
```

---

## Connection to the Web App

These notebooks and the web app (`../celesta/`) share the same:

| What | How |
|---|---|
| **Model architecture** | `BalancedXGBClassifier`, 3-model ensemble — defined identically in `../celesta/model_utils.py` and inline in each notebook |
| **Feature engineering** | All 12 features — same formulas in `../celesta/train_model.py` and notebook Section 6 |
| **Training data** | Both pull from the same NASA Exoplanet Archive URL (notebook downloads live; web app uses `../celesta/data/koi_stripped.csv`) |
| **Model** | The notebook can export `model.joblib` to `../celesta/model.joblib` for the web app to serve |

### Export a freshly trained model to the web app

At the bottom of `Celesta_Submission.ipynb`, after training, add:

```python
import joblib
joblib.dump(ensemble, '../celesta/model.joblib')
print("Model saved to celesta/model.joblib — web app will pick it up automatically.")
```

Then restart the Flask server:

```bash
cd ../celesta && python3 app.py
```

---

## What `Celesta_Submission.ipynb` Covers

| Section | Content |
|---|---|
| 1 | Setup — packages, dark space plot theme |
| 2 | The problem — Kepler, transit photometry, three classes |
| 3 | Data download — live from NASA Exoplanet Archive API |
| 4 | EDA — class balance, missingness, KDE plots, correlation heatmap, boxplots |
| 5 | Missing values & class imbalance — strategy and visualisation |
| 6 | Feature engineering — 12 physics-motivated features with formulas |
| 7 | Model architecture — individual vs ensemble comparison |
| 8 | Training — full ensemble fit |
| 9 | Evaluation — confusion matrix, ROC curves, PR curves, calibration |
| 10 | Cross-validation — 5-fold stratified CV |
| 11 | SHAP — global bar chart, beeswarm, per-class walkthrough |
| 12 | Written summary — 800+ words addressing all judging criteria |
| 13 | Results table & conclusion |

---

*Part of the Celesta project — developed by Srinath Vatchavari Venkateshan*
