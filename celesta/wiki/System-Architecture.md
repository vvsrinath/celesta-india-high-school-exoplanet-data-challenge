# System Architecture

Celesta is built as three completely independent runtimes that share only the trained model file. You can run any one without the others.

**Author:** Srinath V Venkateshan

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Training Pipeline](#2-training-pipeline)
3. [Inference Pipeline (Web Server)](#3-inference-pipeline-web-server)
4. [Explore Section — External API Chain](#4-explore-section--external-api-chain)
5. [Docker Multi-Stage Build](#5-docker-multi-stage-build)
6. [Frontend Architecture](#6-frontend-architecture)
7. [Cache Strategy](#7-cache-strategy)
8. [Data Flow Diagram — End to End](#8-data-flow-diagram--end-to-end)

---

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CELESTA SYSTEM                               │
│                                                                     │
│  ┌─────────────────────────┐   ┌───────────────────────────────┐   │
│  │   TRAINING PIPELINE     │   │      FLASK WEB SERVER         │   │
│  │   (train_model.py)      │   │      (app.py)                 │   │
│  │                         │   │                               │   │
│  │  koi_stripped.csv       │   │  GET  /          → HTML       │   │
│  │       │                 │   │  GET  /api/stats → JSON       │   │
│  │       ▼                 │   │  POST /api/predict → JSON     │   │
│  │  Feature Engineering    │   │  GET  /api/explore/search     │   │
│  │  (27 raw → 35 total)    │   │  GET  /api/explore/details    │   │
│  │       │                 │   │        │                      │   │
│  │       ▼                 │   │        ▼                      │   │
│  │  VotingClassifier       │──►│  model.joblib  (49 MB)        │   │
│  │  ├ XGBoost              │   │  data/stats.json              │   │
│  │  ├ HistGradBoost        │   │                               │   │
│  │  └ RandomForest         │   │  Wikipedia REST API  ──────►  │   │
│  │       │                 │   │  NASA Images API    ──────►   │   │
│  │       ▼                 │   │  Google CSE (opt.)  ──────►   │   │
│  │  model.joblib           │   └───────────────────────────────┘   │
│  │  data/stats.json        │                                        │
│  └─────────────────────────┘   ┌───────────────────────────────┐   │
│                                 │   GITHUB PAGES STATIC DEMO   │   │
│                                 │   (docs/index.html)          │   │
│                                 │                               │   │
│                                 │   No Python. No Flask.        │   │
│                                 │   Stats baked in at build.    │   │
│                                 │   Live prediction needs       │   │
│                                 │   a running Flask server.     │   │
│                                 └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

The three runtimes communicate through **files on disk** (`model.joblib`, `stats.json`) — not through network calls or shared memory. This means:

- The training pipeline runs completely offline, produces two files, then exits.
- The Flask server reads those files at startup and never touches the training code again.
- The GitHub Pages demo embeds the stats at build time and requires no backend at all.

---

## 2. Training Pipeline

`train_model.py` is a single script that runs once. It reads the raw CSV, engineers features, trains the ensemble, evaluates it, computes SHAP values, and writes the two output files.

```
koi_stripped.csv  (1.86 MB — 9,564 rows × 27 raw feature columns + target)
        │
        ▼ pd.read_csv()
   Raw DataFrame
        │
        ▼ add_features()
   Feature Engineering — 8 derived columns added:
   ┌──────────────────────────────────────────────────────────────────┐
   │  single_multi_ratio      = max_sngle_ev / (max_mult_ev + ε)    │
   │  duration_period_ratio   = duration / (period × 24 + ε)        │
   │  log_period              = log1p(period.clip(0))               │
   │  log_depth               = log1p(depth.clip(0))                │
   │  log_snr                 = log1p(model_snr.clip(0))            │
   │  stellar_density_proxy   = smass / (srad³ + ε)                 │
   │  impact_ror_ratio        = impact / (ror + ε)                  │
   │  expected_duration_ratio = duration / T_expected               │
   │    where T_expected = P×24/π × ror/dor × √(1−impact²)         │
   └──────────────────────────────────────────────────────────────────┘
        │
        ▼ LabelEncoder.fit_transform()
   y: CANDIDATE→0, CONFIRMED→1, FALSE POSITIVE→2
        │
        ▼ train_test_split(test_size=0.2, stratify=y, random_state=42)
   X_train (7,651 rows) / X_test (1,913 rows)
        │
        ├─► BalancedXGBClassifier  (model_utils.py)
        │     n_estimators=500, max_depth=6, learning_rate=0.04
        │     subsample=0.8, colsample_bytree=0.8
        │     reg_alpha=0.1, reg_lambda=1.0, objective="multi:softprob"
        │     tree_method="hist"  → native NaN handling, fast histogram splits
        │     On every fit(): sample_weight = compute_sample_weight("balanced", y)
        │
        ├─► HistGradientBoostingClassifier
        │     max_iter=500, max_depth=8, learning_rate=0.04
        │     min_samples_leaf=20, l2_regularization=0.1
        │     class_weight="balanced"
        │     Native NaN handling (missing values treated as a separate branch)
        │
        └─► Pipeline(SimpleImputer(median) → RandomForestClassifier)
              RF: n_estimators=400, max_depth=16, max_features="sqrt"
              class_weight="balanced_subsample"
              Imputer: median per column — RF cannot handle NaN directly
                        │
                        ▼
              VotingClassifier(voting="soft", n_jobs=1)
              ┌─────────────────────────────────────────┐
              │  p_final[c] = (p_xgb[c] + p_hgb[c]     │
              │               + p_rf[c]) / 3            │
              │  prediction = argmax(p_final)            │
              └─────────────────────────────────────────┘
                        │
        ┌───────────────┴────────────────────────────────────────┐
        │                                                        │
        ▼                                                        ▼
5-fold stratified CV                            Test set evaluation
StratifiedKFold(n_splits=5)                     accuracy_score()
cross_val_score(f1_macro)                       f1_score(macro, weighted)
→ 78.65% ± 0.71%                               confusion_matrix()
                                                classification_report()
                        │
                        ▼
              SHAP TreeExplainer(xgb_sub_model)
              Background: 500-sample random subset of X_train
              mean_shap[feature] = mean(|shap_values|) across classes and samples
              Top 15 features ranked and stored
                        │
                        ▼
              joblib.dump({                        data/stats.json
                "pipeline": VotingClassifier,      {accuracy, macro_f1, weighted_f1,
                "label_encoder": LabelEncoder,      cv_macro_f1_mean, cv_macro_f1_std,
                "features": list[35],               per_class, confusion_matrix,
                "engineered": list[8]               feature_importance[15], ...}
              }, "model.joblib")
```

### Why ε = 1e-9?

Every engineered feature that divides by another feature uses `+ 1e-9` in the denominator. The KOI table has rows where `koi_max_mult_ev = 0`, `koi_ror = 0`, or `koi_dor = 0`. Without epsilon, these produce `inf` or `NaN`, which propagate silently through the pipeline and produce garbage predictions. Epsilon is small enough to have no measurable effect on normal rows.

---

## 3. Inference Pipeline (Web Server)

The Flask server loads the model once, caches it in a module-level variable, and applies an identical feature engineering step to every incoming request.

```
POST /api/predict
Body: {"koi_period": 2.2, "koi_prad": 1.5, "koi_max_mult_ev": 435, ...}
        │
        ▼ request.get_json(force=True)
   dict or error 400
        │
        ▼ for feat in raw_features (27 columns):
   ┌───────────────────────────────────────────────────────────┐
   │  val = data.get(feat)                                    │
   │  if val is None or val == "":                            │
   │      row[feat] = np.nan   ← recorded in missing_fields   │
   │  else:                                                   │
   │      row[feat] = float(val)   ← ValueError → NaN         │
   └───────────────────────────────────────────────────────────┘
        │
        ▼ pd.DataFrame([row], columns=raw_features)
   Single-row DataFrame (1 × 27)
        │
        ▼ _add_features(df)
   Same 8 formulas as train_model.py (must stay in sync)
   Single-row DataFrame (1 × 35)
        │
        ▼ X = df[features]   (column order matters — must match training order)
        │
        ├─► VotingClassifier.predict(X)       → [int]   e.g. [1]
        └─► VotingClassifier.predict_proba(X) → [[0.05, 0.87, 0.08]]
        │
        ▼ LabelEncoder.classes_[1] → "CONFIRMED"
        │
        ▼ return jsonify({
              "prediction":    "CONFIRMED",
              "confidence":    0.87,
              "probabilities": {"CANDIDATE": 0.05, "CONFIRMED": 0.87, "FALSE POSITIVE": 0.08},
              "missing_fields": ["koi_srho", "koi_time0bk", ...]
          })
```

**Model cache:** `_model_cache` and `_stats_cache` are module-level globals initialised to `None`. `load_model()` populates them on first call (joblib.load takes ~3 s) and returns the cached objects on every subsequent call.

**Thread safety:** gunicorn runs 2 workers with `--timeout 120`. Each worker process has its own `_model_cache` — no shared memory, no locking needed.

---

## 4. Explore Section — External API Chain

```
User types in the search box
        │ (debounced 300 ms)
        ▼
GET /api/explore/search?q={query}
        │
        └─► Wikipedia OpenSearch API
              https://en.wikipedia.org/w/api.php
                ?action=opensearch&search={q}&limit=8&namespace=0&format=json
              Returns: [query, [titles], [descriptions], [urls]]
              Cache: lru_cache(maxsize=128) — keyed by exact query string
              Timeout: 6 s
              Response: [{title, description}, ...]
                        │
                        ▼
              Dropdown rendered in the browser

User clicks a search result
        │
        ▼
GET /api/explore/details?name={wikipedia_article_name}
        │
        ├─► Wikipedia REST Summary
        │     https://en.wikipedia.org/api/rest_v1/page/summary/{slug}
        │     Returns: title, description, extract, thumbnail, originalimage,
        │               content_urls.desktop.page
        │     Cache: lru_cache(maxsize=128) — keyed by URL-encoded slug
        │     Timeout: 6 s
        │
        ├─► NASA Images API
        │     https://images-api.nasa.gov/search
        │       ?q={name}&media_type=image&page_size=10
        │     Iterates over 10 results, calls _nasa_relevant(title, query):
        │       PASS: query term appears in image title
        │             AND title contains no noise keywords
        │             (launch, rollout, astronaut, crew, rocket, ...)
        │     Returns: first href that passes the filter, or None
        │     Cache: lru_cache(maxsize=128)
        │     Timeout: 6 s
        │
        └─► Google Custom Search API  (optional — only if env vars set)
              https://www.googleapis.com/customsearch/v1
                ?key={KEY}&cx={CX}&q={name}+space+telescope
                &searchType=image&num=1&imgSize=large
              Timeout: 4 s
              Returns: first image link, or None if request fails
                        │
                        ▼
              Image priority: NASA → Wikipedia full-size → Google
              ┌────────────────────────────────────────┐
              │  image = nasa_img or wiki_full or       │
              │          google_img                     │
              │  images = [nasa_img, wiki_full,         │
              │            google_img] (deduplicated,   │
              │            max 3)                       │
              └────────────────────────────────────────┘
                        │
                        ▼
              JSON response to browser:
              {title, description, extract, image, images[3], wiki_link}
```

**Why lru_cache instead of Redis/disk cache?** This is a single-process development server. `lru_cache` is zero-config and adds no dependencies. The cache survives for the lifetime of the process — a server restart clears it, which is acceptable since the data is not time-critical. If the app scaled to multiple gunicorn workers, a shared Redis cache would be needed.

---

## 5. Docker Multi-Stage Build

```
docker build -t celesta .
        │
        ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 1: trainer  (python:3.11-slim + build-essential)    │
│                                                            │
│  Installed packages:                                       │
│    flask, gunicorn, requests, numpy, pandas                │
│    scikit-learn 1.9, joblib, xgboost ≥ 2.0                │
│    shap ≥ 0.44, imbalanced-learn ≥ 0.12, lightgbm         │
│    build-essential (C compiler for xgboost wheel)          │
│  Total: ~ 1.2 GB in this stage                             │
│                                                            │
│  Files copied in:                                          │
│    train_model.py, model_utils.py                          │
│    data/koi_stripped.csv                                   │
│                                                            │
│  RUN python3 train_model.py                                │
│    → /build/model.joblib     (49 MB)                       │
│    → /build/data/stats.json  (7 KB)                        │
│    (Takes ~3 min on 2 vCPUs)                               │
└────────────────────────────────────────────────────────────┘
        │
        │  COPY --from=trainer /build/model.joblib     .
        │  COPY --from=trainer /build/data/stats.json  data/stats.json
        ▼
┌────────────────────────────────────────────────────────────┐
│  Stage 2: runtime  (python:3.11-slim, fresh base)          │
│                                                            │
│  Installed packages:                                       │
│    flask, gunicorn, requests, numpy, pandas                │
│    scikit-learn 1.9, joblib, xgboost ≥ 2.0                │
│  NOT installed: shap, imbalanced-learn, lightgbm           │
│  NOT installed: build-essential                            │
│  Total: ~ 400–500 MB                                       │
│                                                            │
│  Files copied in:                                          │
│    app.py, model_utils.py                                  │
│    templates/, static/                                     │
│    model.joblib    ← from trainer stage                    │
│    data/stats.json ← from trainer stage                    │
│                                                            │
│  EXPOSE 5000                                               │
│  ENV GOOGLE_CSE_CX=""                                      │
│  ENV GOOGLE_CSE_API_KEY=""                                 │
│                                                            │
│  CMD gunicorn --bind 0.0.0.0:5000                          │
│               --workers 2 --timeout 120 app:app            │
└────────────────────────────────────────────────────────────┘
```

**Why must xgboost be in the runtime stage?** `model.joblib` was pickled with XGBoost as a dependency. Unpickling it requires XGBoost to be installed in the runtime image, even though no training happens there. SHAP and imbalanced-learn, however, are only used during training and are safe to exclude.

---

## 6. Frontend Architecture

The entire frontend is a single HTML file (`templates/index.html`) rendered by Flask via Jinja2, plus two static files (`style.css`, `app.js`). There is no JavaScript framework, no bundler, and no build step.

```
Browser requests GET /
        │
        ▼ Flask renders templates/index.html
          Jinja2 injects: stats.accuracy, stats.macro_f1, stats.classes,
                          stats.per_class, stats.feature_importance,
                          stats.confusion_matrix, stats.n_total
        │
        ▼ Browser parses HTML, loads:
          static/css/style.css      (dark space theme via CSS custom properties)
          Three.js r134              (CDN: cdn.jsdelivr.net)
          static/js/app.js           (all application logic)
        │
        ▼ app.js initialises in sections:

        ┌── Three.js WebGL Scene ──────────────────────────────┐
        │  requestIdleCallback → initThree()                   │
        │    renderer: WebGLRenderer, antialias=false           │
        │    pixelRatio: min(devicePixelRatio, 1.2)            │
        │    Stars: 1200 particles, BufferGeometry+PointsMat   │
        │    Skybox: SphereGeometry(950), BackSide texture      │
        │    Planet: SphereGeometry(90), textured, pos(320,-60)│
        │    Orbs: 3× low-opacity glow spheres                 │
        │    Loop: planet.rotation.y += 0.002/frame            │
        │           camera offset ← mouse position (lerped)    │
        └──────────────────────────────────────────────────────┘

        ┌── Explore Section ───────────────────────────────────┐
        │  input event (debounced)                             │
        │    → fetch /api/explore/search?q=...                 │
        │    → render dropdown with up to 8 results            │
        │  click on result                                     │
        │    → fetch /api/explore/details?name=...             │
        │    → render title, description, extract, image       │
        └──────────────────────────────────────────────────────┘

        ┌── Performance Dashboard ─────────────────────────────┐
        │  Confusion matrix: 3×3 CSS Grid, values from Jinja2  │
        │  Per-class bars: CSS width transitions               │
        │  SHAP chart: sorted bars, CSS width from SHAP values │
        │  Counters: IntersectionObserver → animated count-up  │
        └──────────────────────────────────────────────────────┘

        ┌── Classifier Form ───────────────────────────────────┐
        │  submit event                                        │
        │    → collect 27 visible inputs (skip empty = NaN)   │
        │    → fetch POST /api/predict                         │
        │    → render: label badge, confidence, prob bars      │
        │    → show missing_fields count if > 0               │
        └──────────────────────────────────────────────────────┘
```

**Design system:** All colours, radii, and spacings are CSS custom properties on `:root`. The dark space palette (`#0a0b14` background, `#6366f1` primary accent, `#06b6d4` secondary accent) is consistent across all sections. No CSS framework is used — layout is Grid and Flexbox throughout.

---

## 7. Cache Strategy

| Layer | Mechanism | Duration | Reason |
|---|---|---|---|
| Browser / proxy responses | `Cache-Control: no-cache, no-store, must-revalidate` via `@app.after_request` | 0 s — always revalidate | Predictions and explore results must always be fresh. No stale data served. |
| Flask static files | `app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0` | 0 s | Overrides Flask's default 12-hour cache for `static/` files during development. |
| GitHub Pages CSS/JS | `?v=2` query string on `<link>` and `<script>` tags | Indefinite until version bump | Fastly CDN (GitHub Pages' CDN) caches by full URL including query string. Bumping `?v=2` → `?v=3` forces a new cache entry. |
| Wikipedia search results | Python `lru_cache(maxsize=128)` per process | Process lifetime | Wikipedia search results for astronomy terms change rarely. Avoiding repeated HTTP calls reduces latency from ~400 ms to ~0 ms on repeated queries. |
| NASA image results | Python `lru_cache(maxsize=128)` per process | Process lifetime | Same reasoning. NASA image URLs for a given object are stable. |
| Model artefacts | Module-level `_model_cache` dict | Process lifetime | joblib.load of a 49 MB file takes ~3 s. Loading once at startup (or on first request) and caching permanently keeps every subsequent request at <50 ms. |
| Prediction responses | No caching | 0 s | Different input feature values must always reach the model. |

---

## 8. Data Flow Diagram — End to End

```
                        NASA Exoplanet Archive
                               │
                      download (once)
                               │
                               ▼
                     koi_stripped.csv (1.86 MB)
                               │
                    python3 train_model.py
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
              model.joblib          data/stats.json
              (49 MB, gitignored)   (7 KB, tracked)
                    │                     │
         ┌──────────┴──────────┐          │
         │                     │          │
         ▼                     ▼          ▼
  python3 app.py          docker build  docs/index.html
  (Flask server)          (retrains     (GitHub Pages)
         │                 in stage 1)
         │
  ┌──────┴────────────────────────────────────┐
  │              User's browser               │
  │                                           │
  │  GET /         → full page HTML           │
  │  GET /api/stats → metrics JSON            │
  │  POST /api/predict → prediction JSON      │
  │  GET /api/explore/* → Wikipedia/NASA JSON │
  └───────────────────────────────────────────┘
```
