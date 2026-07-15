"""
app.py — Celesta

Flask web server for the Celesta exoplanet classifier.

Routes
------
GET  /                      Renders the main HTML page.
GET  /api/stats             Returns training metrics from stats.json.
POST /api/predict           Classifies a single KOI from raw transit/stellar inputs.
GET  /api/explore/search    Searches Wikipedia for astronomy topics.
GET  /api/explore/details   Returns Wikipedia summary + NASA/Google images for a topic.

The model is loaded once at startup and kept in a module-level cache.
All responses disable browser caching so predictions are always fresh.

Run:
    python3 app.py
"""

import json
import os
import pathlib
import urllib.parse
import numpy as np
import joblib
import requests as http_requests
from functools import lru_cache
from flask import Flask, render_template, request, jsonify
from model_utils import BalancedXGBClassifier  # noqa: F401 - needed so joblib can unpickle the model

ROOT = pathlib.Path(__file__).parent
MODEL_PATH = ROOT / "model.joblib"
STATS_PATH = ROOT / "data" / "stats.json"

app = Flask(__name__)
# never serve stale content — every response must be revalidated
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def set_cache_headers(response):
    """Attach no-cache headers to every response.

    Browsers and proxies must revalidate before serving any cached copy.
    This matters most for /api/predict — a user changing feature values must
    always get a fresh prediction, never a cached one from a previous request.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Vary"] = "Accept-Encoding"
    return response


_model_cache = None
_stats_cache = None


def load_model():
    """Load model.joblib and stats.json, caching both after the first call.

    Returns
    -------
    artefacts : dict
        Keys: ``pipeline`` (VotingClassifier), ``label_encoder`` (LabelEncoder),
        ``features`` (list of 35 feature names), ``engineered`` (list of 8
        engineered feature names).
    stats : dict
        Training metrics written by train_model.py.

    Raises
    ------
    FileNotFoundError
        If model.joblib does not exist.  Run ``python3 train_model.py`` first.
    """
    global _model_cache, _stats_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("model.joblib not found - run train_model.py first")
        _model_cache = joblib.load(MODEL_PATH)
    if _stats_cache is None:
        _stats_cache = json.loads(STATS_PATH.read_text())
    return _model_cache, _stats_cache


def _add_features(df):
    """Apply the 8 engineered features to a single-row DataFrame at inference time.

    This mirrors ``add_features()`` in train_model.py exactly.  Any change to
    the feature logic must be applied in both places, then the model retrained.

    Parameters
    ----------
    df : pd.DataFrame
        One-row DataFrame containing the raw KOI columns.  Missing columns are
        treated as NaN — the ensemble handles them via native NaN support
        (XGBoost, HistGradientBoosting) or median imputation (RandomForest).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with 8 additional columns appended.
    """
    d = df.copy()
    eps = 1e-9
    d["single_multi_ratio"]      = d.get("koi_max_sngle_ev", float("nan")) / (d.get("koi_max_mult_ev", float("nan")) + eps)
    d["duration_period_ratio"]   = d.get("koi_duration", float("nan")) / (d.get("koi_period", float("nan")) * 24.0 + eps)
    d["log_period"]              = np.log1p(np.clip(d.get("koi_period", float("nan")), 0, None))
    d["log_depth"]               = np.log1p(np.clip(d.get("koi_depth", float("nan")), 0, None))
    d["log_snr"]                 = np.log1p(np.clip(d.get("koi_model_snr", float("nan")), 0, None))
    d["stellar_density_proxy"]   = d.get("koi_smass", float("nan")) / (d.get("koi_srad", float("nan")) ** 3 + eps)
    d["impact_ror_ratio"]        = d.get("koi_impact", float("nan")) / (d.get("koi_ror", float("nan")) + eps)
    expected = (
        d.get("koi_period", float("nan")) * 24.0 / np.pi
        * d.get("koi_ror", float("nan"))
        / (d.get("koi_dor", float("nan")) + eps)
        * np.sqrt(np.clip(1 - d.get("koi_impact", float("nan")) ** 2, 0, None))
    )
    d["expected_duration_ratio"] = d.get("koi_duration", float("nan")) / (expected + eps)
    return d


@app.route("/")
def index():
    """Render the main page, injecting live model stats into the template."""
    _, stats = load_model()
    return render_template("index.html", stats=stats)


@app.route("/api/stats")
def api_stats():
    """Return training metrics as JSON.

    Response shape is documented in the wiki API Reference → GET /api/stats.
    """
    _, stats = load_model()
    return jsonify(stats)


@app.route("/api/predict", methods=["POST"])
def predict():
    """Classify a single KOI from raw transit and stellar measurements.

    Accepts a JSON object with any subset of the 27 raw feature fields.
    Missing or empty fields are treated as NaN.  Engineered features are
    computed server-side and must NOT be sent by the caller.

    Returns
    -------
    JSON with keys:
        prediction    : str   — "CANDIDATE", "CONFIRMED", or "FALSE POSITIVE"
        confidence    : float — probability of the predicted class (0–1)
        probabilities : dict  — soft probability for each class
        missing_fields: list  — raw fields that were absent/empty (treated as NaN)

    Status codes:
        200  — success
        400  — request body is not a JSON object
        500  — model not loaded (run train_model.py first)
    """
    import pandas as pd

    artefacts, _ = load_model()
    clf = artefacts["pipeline"]
    le = artefacts["label_encoder"]
    features = artefacts["features"]
    engineered = artefacts.get("engineered", [])

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "expected a JSON object"}), 400

    raw_features = [f for f in features if f not in engineered]
    row = {}
    missing = []
    for feat in raw_features:
        val = data.get(feat)
        if val is None or val == "":
            row[feat] = np.nan
            missing.append(feat)
        else:
            try:
                row[feat] = float(val)
            except (ValueError, TypeError):
                row[feat] = np.nan

    X = pd.DataFrame([row], columns=raw_features)
    if engineered:
        X = _add_features(X)
        X = X[features]

    pred_idx = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = le.classes_[pred_idx]

    return jsonify({
        "prediction": label,
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(le.classes_, proba)},
        "missing_fields": missing,
    })


# ── Explore section ──────────────────────────────────────────────────────────
# Fetches astronomy information from Wikipedia and NASA Images.
# Responses are cached in-process with lru_cache; the cache is discarded on
# server restart.  This is intentional — data should stay fresh between runs.

_WIKI_HEADERS = {"User-Agent": "Celesta/1.0 (vvsrinath0@gmail.com)"}
_TIMEOUT = 6

_GOOGLE_CX  = os.environ.get("GOOGLE_CSE_CX")
_GOOGLE_KEY = os.environ.get("GOOGLE_CSE_API_KEY")


def _google_image(q):
    """Return the first Google Custom Search image URL for query ``q``.

    Returns None if the API credentials are not configured or the request fails.
    Both GOOGLE_CSE_CX and GOOGLE_CSE_API_KEY must be set as environment
    variables.
    """
    if not _GOOGLE_CX or not _GOOGLE_KEY:
        return None
    try:
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?key={_GOOGLE_KEY}&cx={_GOOGLE_CX}"
            f"&q={urllib.parse.quote(q + ' space telescope')}"
            "&searchType=image&num=1&imgSize=large"
        )
        items = http_requests.get(url, timeout=4).json().get("items", [])
        if items:
            return items[0].get("link")
    except Exception:
        pass
    return None


# NASA image search sometimes returns rocket launch photos for star names,
# so filter those out
_NASA_NOISE = (
    "rollout", "launch", "apollo", "gemini", "artemis", "rocket",
    "kennedy space center", "ksc-", "astronaut", "crew", "mission patch",
    "underwater", "training", "simulator", "press conference", "portrait",
    "dawn program", "vehicle", "test flight", "assembly building",
    "spacecraft integration", "payload processing", "expedition",
    "flight readiness", "media day", "recovery ship",
)


def _nasa_relevant(title, query):
    """Return True if the NASA image title is topically related to the query.

    NASA's image API often surfaces rocket launch or crew photos when searching
    for star names (e.g. "Gemini" matches the Gemini programme, not the
    constellation).  This filter requires the query term to appear in the title
    and rejects titles containing known off-topic keywords.
    """
    t, q = title.lower(), query.lower()
    return q in t and not any(w in t for w in _NASA_NOISE)


@lru_cache(maxsize=128)
def _wiki_search(q):
    """Search Wikipedia and return up to 8 title/description pairs.

    Uses the Wikipedia OpenSearch API.  Results are cached per unique query
    string.

    Parameters
    ----------
    q : str
        Search query.

    Returns
    -------
    list of dict with keys ``title`` and ``description``.
    """
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={urllib.parse.quote(q)}"
        "&limit=8&namespace=0&format=json"
    )
    data = http_requests.get(url, headers=_WIKI_HEADERS, timeout=_TIMEOUT).json()
    titles = data[1] if len(data) > 1 else []
    descriptions = data[2] if len(data) > 2 else []
    return [{"title": t, "description": d} for t, d in zip(titles, descriptions)]


@lru_cache(maxsize=128)
def _wiki_summary(slug):
    """Fetch the Wikipedia REST summary for a page slug.

    Parameters
    ----------
    slug : str
        URL-encoded Wikipedia page title (spaces replaced with underscores).

    Returns
    -------
    dict
        Wikipedia REST API summary object, or ``{}`` if the page does not exist.
    """
    r = http_requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}",
        headers=_WIKI_HEADERS, timeout=_TIMEOUT
    )
    return r.json() if r.status_code == 200 else {}


@lru_cache(maxsize=128)
def _nasa_image(q):
    """Return the first relevant NASA Images API image URL for query ``q``.

    Iterates over results until one passes the ``_nasa_relevant`` filter.
    Returns None if no relevant image is found or the request fails.
    """
    try:
        url = f"https://images-api.nasa.gov/search?q={urllib.parse.quote(q)}&media_type=image&page_size=10"
        items = http_requests.get(url, headers=_WIKI_HEADERS, timeout=_TIMEOUT).json().get("collection", {}).get("items", [])
        for item in items:
            title = (item.get("data") or [{}])[0].get("title", "")
            if not _nasa_relevant(title, q):
                continue
            links = item.get("links", [])
            if links:
                return links[0].get("href")
    except Exception:
        pass
    return None


@app.route("/api/explore/search")
def explore_search():
    """Search Wikipedia for astronomy topics.

    Query parameters
    ----------------
    q : str
        Search query (minimum 2 characters).

    Returns
    -------
    JSON array of up to 8 {title, description} objects, or [] if q is too short.
    502 on Wikipedia API failure.
    """
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify([])
    try:
        return jsonify(_wiki_search(q))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/explore/details")
def explore_details():
    """Return Wikipedia summary and images for a named astronomy object.

    Fetches the Wikipedia REST summary, then augments it with:
    - NASA Images API (filtered to exclude off-topic results)
    - Wikipedia original image
    - Google Custom Search image (only if GOOGLE_CSE_* env vars are set)

    Image priority: NASA > Wikipedia > Google.

    Query parameters
    ----------------
    name : str
        Wikipedia article name, e.g. "51 Pegasi b".

    Returns
    -------
    JSON with keys: title, description, extract, image, images (up to 3), wiki_link.
    400 if name is missing.  502 on upstream API failure.
    """
    name = request.args.get("name", "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    try:
        slug = urllib.parse.quote(name.replace(" ", "_"))
        wiki = _wiki_summary(slug)

        title       = wiki.get("title", name)
        extract     = wiki.get("extract", "No description found.")
        wiki_link   = (wiki.get("content_urls") or {}).get("desktop", {}).get("page", "")
        description = wiki.get("description", "")

        wiki_thumb  = (wiki.get("thumbnail") or {}).get("source")
        wiki_full   = (wiki.get("originalimage") or {}).get("source") or wiki_thumb
        nasa_img    = _nasa_image(name)
        google_img  = _google_image(name)

        # prefer NASA > Wikipedia > Google for image quality
        image = nasa_img or wiki_full or google_img
        images = []
        for img in [nasa_img, wiki_full, google_img]:
            if img and img not in images:
                images.append(img)

        return jsonify({
            "title": title,
            "description": description,
            "extract": extract,
            "image": image,
            "images": images[:3],
            "wiki_link": wiki_link,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
