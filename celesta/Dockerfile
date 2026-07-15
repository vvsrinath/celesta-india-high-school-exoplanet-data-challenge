# Celesta — multi-stage Docker build
#
# Stage 1 (trainer): installs ALL packages (ML + web), trains the model.
#                    Produces model.joblib and data/stats.json.
# Stage 2 (runtime): installs only web/runtime packages (no shap, no
#                    imbalanced-learn). Copies the trained artefacts from
#                    stage 1 and serves via gunicorn.
#
# Build:  docker build -t celesta .
# Run:    docker run -p 5000:5000 celesta
#         docker run -p 5000:5000 -e GOOGLE_CSE_CX=xxx -e GOOGLE_CSE_API_KEY=yyy celesta

# ── Stage 1: train the model ─────────────────────────────────────────────────
FROM python:3.11-slim AS trainer

WORKDIR /build

# build-essential is needed to compile any C extensions in the wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# install runtime deps first, then training-only extras on top
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-train.txt

COPY train_model.py model_utils.py ./
COPY data/koi_stripped.csv data/koi_stripped.csv

RUN python3 train_model.py

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# xgboost is needed at runtime to unpickle model.joblib — it must be present
# even though no training happens here. shap and imbalanced-learn are NOT needed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py model_utils.py ./
COPY templates/ templates/
COPY static/ static/

# trained artefacts from stage 1
COPY --from=trainer /build/model.joblib .
COPY --from=trainer /build/data/stats.json data/stats.json

EXPOSE 5000

ENV GOOGLE_CSE_CX=""
ENV GOOGLE_CSE_API_KEY=""

# 2 workers handles concurrent users; 120 s timeout covers cold model load (~3 s)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
