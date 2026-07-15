# 🌌 Celesta — Kepler Exoplanet Classifier

**Developed by Srinath Vatchavari Venkateshan**  
India High School Exoplanet Data Challenge

---

This repository has two top-level folders:

| Folder | What's inside |
|---|---|
| **[`celesta/`](celesta/)** | Flask web app — model training, prediction API, frontend · **start here for the running app** |
| **[`colab/`](colab/)** | Google Colab notebooks — full ML pipeline, EDA, SHAP · **start here for the submission notebook** |

---

## Quick Start

### Run the web app

```bash
cd celesta
pip install -r requirements.txt
python3 app.py          # → http://localhost:5000
```

### Open the submission notebook in Colab

See [`colab/README.md`](colab/README.md) for Colab badges and instructions.

---

## Repository Size

| What | Size | In git? |
|---|---|---|
| `celesta/` source | ~3.9 MB | ✅ Yes |
| `colab/` notebooks | ~130 KB | ✅ Yes |
| `celesta/model.joblib` | ~47 MB | ❌ No — gitignored, generated locally |
| **Total tracked** | **< 5 MB** | — |

`model.joblib` is excluded from git (see `celesta/.gitignore`). Run
`python3 celesta/train_model.py` to generate it locally, or use the Docker
multi-stage build which trains it automatically.

---

*Developed by Srinath Vatchavari Venkateshan*
