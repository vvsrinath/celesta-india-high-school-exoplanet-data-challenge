# API Reference

All endpoints are served by the Flask application in `app.py`. The server runs on `http://localhost:5000` by default.

Every response sets `Cache-Control: no-cache, no-store, must-revalidate` so browsers never serve stale predictions.

---

## Table of Contents

1. [GET /](#1-get-)
2. [GET /api/stats](#2-get-apistats)
3. [POST /api/predict](#3-post-apipredict)
4. [GET /api/explore/search](#4-get-apiexploresearch)
5. [GET /api/explore/details](#5-get-apiexploredetails)
6. [Error Format](#6-error-format)
7. [Optional Environment Variables](#7-optional-environment-variables)

---

## 1. `GET /`

Renders the main HTML page.

**Response:** `text/html` — the Jinja2 template `templates/index.html` with the `stats` object injected.

**Side effect:** Loads `model.joblib` and `data/stats.json` into memory on first request (warm-up takes ~2 s).

---

## 2. `GET /api/stats`

Returns the training metrics written by `train_model.py`.

**Response:** `application/json`

```json
{
  "accuracy": 0.8134,
  "macro_f1": 0.7884,
  "weighted_f1": 0.8152,
  "macro_recall": 0.7956,
  "macro_precision": 0.7829,
  "cv_macro_f1_mean": 0.7865,
  "cv_macro_f1_std": 0.0071,
  "n_train": 7651,
  "n_test": 1913,
  "n_total": 9564,
  "classes": ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"],
  "class_counts": {
    "CANDIDATE": 1978,
    "CONFIRMED": 2747,
    "FALSE POSITIVE": 4839
  },
  "confusion_matrix": [[265, 48, 83], [41, 487, 21], [123, 41, 804]],
  "per_class": {
    "CANDIDATE":      {"precision": 0.618, "recall": 0.669, "f1": 0.642, "support": 396},
    "CONFIRMED":      {"precision": 0.845, "recall": 0.887, "f1": 0.866, "support": 549},
    "FALSE POSITIVE": {"precision": 0.885, "recall": 0.831, "f1": 0.857, "support": 968}
  },
  "feature_importance": [["koi_max_mult_ev", 0.548], ...],
  "features": ["koi_period", "koi_time0bk", ...],
  "n_features_raw": 27,
  "n_features_engineered": 8
}
```

**Example:**

```bash
curl http://localhost:5000/api/stats
```

---

## 3. `POST /api/predict`

Classifies a single Kepler Object of Interest.

**Request body:** `application/json` — a flat JSON object with any subset of the 27 raw feature keys listed below. Fields that are omitted or empty string are treated as `NaN`; the model handles missing values internally.

### Input fields

| Field | Description | Units |
|---|---|---|
| `koi_period` | Orbital period | days |
| `koi_time0bk` | Time of first transit | days (BJD − 2 454 833) |
| `koi_impact` | Transit impact parameter | dimensionless (0–1+) |
| `koi_duration` | Transit duration | hours |
| `koi_depth` | Transit depth (flux drop) | ppm |
| `koi_ror` | Planet-to-star radius ratio | dimensionless |
| `koi_srho` | Fitted stellar density | g/cm³ |
| `koi_prad` | Planet radius | Earth radii |
| `koi_sma` | Semi-major axis | AU |
| `koi_incl` | Orbital inclination | degrees |
| `koi_teq` | Equilibrium temperature | K |
| `koi_insol` | Insolation flux | Earth flux |
| `koi_dor` | Semi-major axis / stellar radius | dimensionless |
| `koi_max_sngle_ev` | Maximum single-event statistic | σ |
| `koi_max_mult_ev` | Maximum multiple-event statistic | σ |
| `koi_model_snr` | Transit model signal-to-noise | dimensionless |
| `koi_count` | Number of associated KOIs in the system | integer |
| `koi_num_transits` | Number of observed transits | integer |
| `koi_bin_oedp_sig` | Odd-even depth difference significance | σ |
| `koi_steff` | Stellar effective temperature | K |
| `koi_slogg` | Stellar surface gravity | log₁₀(cm/s²) |
| `koi_smet` | Stellar metallicity | dex |
| `koi_srad` | Stellar radius | solar radii |
| `koi_smass` | Stellar mass | solar masses |
| `ra` | Right ascension | degrees |
| `dec` | Declination | degrees |
| `koi_kepmag` | Kepler magnitude | mag |

> **Note:** The 8 engineered features (`single_multi_ratio`, `duration_period_ratio`, `log_period`, `log_depth`, `log_snr`, `stellar_density_proxy`, `impact_ror_ratio`, `expected_duration_ratio`) are computed server-side from the raw inputs — do not send them.

### Response

```json
{
  "prediction": "CONFIRMED",
  "confidence": 0.9123,
  "probabilities": {
    "CANDIDATE":      0.0412,
    "CONFIRMED":      0.9123,
    "FALSE POSITIVE": 0.0465
  },
  "missing_fields": ["koi_srho", "koi_bin_oedp_sig"]
}
```

| Field | Type | Description |
|---|---|---|
| `prediction` | string | The highest-probability class label |
| `confidence` | float 0–1 | Probability assigned to the predicted class |
| `probabilities` | object | Soft probability for each class |
| `missing_fields` | array | Raw fields that were absent or empty (treated as NaN) |

### Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "koi_period": 2.2047,
    "koi_impact": 0.13,
    "koi_duration": 1.8,
    "koi_depth": 9090.0,
    "koi_ror": 0.097,
    "koi_prad": 1.08,
    "koi_max_mult_ev": 435.0,
    "koi_max_sngle_ev": 27.0,
    "koi_model_snr": 412.0,
    "koi_count": 3,
    "koi_steff": 5750,
    "koi_srad": 1.0,
    "koi_smass": 1.0,
    "koi_smet": 0.05,
    "koi_dor": 6.3,
    "koi_slogg": 4.44
  }'
```

### Error responses

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "expected a JSON object"}` | Body is not a JSON object |
| `500` | Flask default error page | `model.joblib` not found (run `train_model.py` first) |

---

## 4. `GET /api/explore/search`

Searches Wikipedia for astronomy topics. Results are cached in-process with `lru_cache` (up to 128 unique queries).

**Query parameters:**

| Parameter | Required | Description |
|---|---|---|
| `q` | Yes | Search query string (min 2 characters) |

**Response:** `application/json` — array of up to 8 results.

```json
[
  {
    "title": "51 Pegasi b",
    "description": "Extrasolar planet orbiting the Sun-like star 51 Pegasi"
  },
  ...
]
```

Returns `[]` if `q` is missing or shorter than 2 characters.

**Example:**

```bash
curl "http://localhost:5000/api/explore/search?q=hot+jupiter"
```

**Error response:**

| Status | Body | Cause |
|---|---|---|
| `502` | `{"error": "..."}` | Wikipedia API unreachable |

---

## 5. `GET /api/explore/details`

Fetches a Wikipedia summary, image, and NASA image for a named astronomy object. Results are cached per unique name.

**Query parameters:**

| Parameter | Required | Description |
|---|---|---|
| `name` | Yes | Wikipedia article name (e.g. `"51 Pegasi b"`) |

**Response:** `application/json`

```json
{
  "title": "51 Pegasi b",
  "description": "Extrasolar planet orbiting the Sun-like star 51 Pegasi",
  "extract": "51 Pegasi b (abbreviated 51 Peg b) ...",
  "image": "https://upload.wikimedia.org/...",
  "images": [
    "https://images-assets.nasa.gov/...",
    "https://upload.wikimedia.org/...",
    "https://www.googleapis.com/..."
  ],
  "wiki_link": "https://en.wikipedia.org/wiki/51_Pegasi_b"
}
```

| Field | Description |
|---|---|
| `title` | Wikipedia page title |
| `description` | Short description from Wikipedia |
| `extract` | Full introductory text from Wikipedia |
| `image` | Best available image URL (NASA > Wikipedia > Google) |
| `images` | Up to 3 image URLs in preference order |
| `wiki_link` | URL to the full Wikipedia article |

**Image priority:** NASA Images API → Wikipedia full-size image → Google Custom Search (only if `GOOGLE_CSE_CX` and `GOOGLE_CSE_API_KEY` are set).

**Error responses:**

| Status | Body | Cause |
|---|---|---|
| `400` | `{"error": "name required"}` | `name` parameter missing |
| `502` | `{"error": "..."}` | Wikipedia/NASA API unreachable |

---

## 6. Error Format

All API errors return JSON:

```json
{ "error": "<human-readable message>" }
```

---

## 7. Optional Environment Variables

| Variable | Used by | Purpose |
|---|---|---|
| `GOOGLE_CSE_CX` | `/api/explore/details` | Google Custom Search Engine ID — enables Google image fallback |
| `GOOGLE_CSE_API_KEY` | `/api/explore/details` | Google CSE API key — required alongside `GOOGLE_CSE_CX` |

If either variable is unset, Google image search is silently skipped and NASA/Wikipedia images are used instead.
