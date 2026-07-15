"""
train_model.py — Celesta

Trains the exoplanet classifier on the KOI Cumulative dataset and writes:

    model.joblib      — joblib-pickled dict containing the fitted ensemble,
                        label encoder, feature list, and engineered feature list.
                        (~49 MB, gitignored)
    data/stats.json   — accuracy, F1, CV scores, confusion matrix, per-class
                        metrics, and SHAP feature importance (top 15).
    data/koi_stripped.csv  — raw-features-only CSV used by the notebooks.

Run once before starting app.py:
    python3 train_model.py

Re-run any time features or hyperparameters change.
"""

import json
import warnings
import pathlib
import numpy as np
import pandas as pd
import joblib
import shap

from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from model_utils import BalancedXGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

ROOT = pathlib.Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# find the dataset - check a couple of locations
_paths = [DATA_DIR / "koi_stripped.csv", ROOT / "KOI_Cumulative_clean.csv", ROOT.parent / "KOI_Cumulative_clean.csv"]
DATA_PATH = next((p for p in _paths if p.exists()), _paths[0])

TARGET = "koi_disposition"

# koi_sage, koi_model_dof, koi_model_chisq are 100% missing in this dataset so just drop them
RAW_FEATURES = [
    "koi_period", "koi_time0bk", "koi_impact", "koi_duration",
    "koi_depth", "koi_ror", "koi_srho", "koi_prad", "koi_sma",
    "koi_incl", "koi_teq", "koi_insol", "koi_dor",
    "koi_max_sngle_ev", "koi_max_mult_ev", "koi_model_snr",
    "koi_count", "koi_num_transits", "koi_bin_oedp_sig",
    "koi_steff", "koi_slogg", "koi_smet",
    "koi_srad", "koi_smass",
    "ra", "dec", "koi_kepmag",
]

ENGINEERED = [
    "single_multi_ratio", "duration_period_ratio",
    "log_period", "log_depth", "log_snr",
    "stellar_density_proxy", "impact_ror_ratio", "expected_duration_ratio",
]


def add_features(df):
    """Compute 8 physics-motivated features from the raw KOI columns.

    All features use a small epsilon (1e-9) in denominators to avoid
    division by zero on edge-case rows.

    Features added
    --------------
    single_multi_ratio
        koi_max_sngle_ev / koi_max_mult_ev.  A high ratio means a large
        one-off event relative to the repeating-transit stack — a classic
        false-positive signature.
    duration_period_ratio
        koi_duration / (koi_period * 24).  Encodes the transit chord
        length; tied to stellar density through Kepler's third law.
    log_period, log_depth, log_snr
        Log1p transforms of three right-skewed features.  Makes their
        distributions closer to Gaussian, which benefits the gradient
        boosting base learners.
    stellar_density_proxy
        koi_smass / koi_srad³.  Proportional to mean stellar density;
        provides an independent check on the transit geometry.
    impact_ror_ratio
        koi_impact / koi_ror.  Values near 1 indicate a grazing transit,
        which is a common configuration for eclipsing binary false positives.
    expected_duration_ratio
        Observed duration divided by the Keplerian predicted duration
        (T = P*24/π * Rp/R★ / (a/R★) * √(1-b²)).  Large deviations point
        to orbital eccentricity or a blended background binary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least the raw KOI columns used below.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with 8 additional columns.
    """
    d = df.copy()
    eps = 1e-9

    # ratio of single to multi event stat - when this is high it usually means
    # a one-off event rather than a repeating transit (strong FP indicator)
    d["single_multi_ratio"] = d["koi_max_sngle_ev"] / (d["koi_max_mult_ev"] + eps)

    # how long the transit is relative to the orbital period
    # this encodes the transit chord length and ties into stellar density via Kepler III
    d["duration_period_ratio"] = d["koi_duration"] / (d["koi_period"] * 24.0 + eps)

    # periods, depths and snr are all heavily right-skewed so log transform helps
    d["log_period"] = np.log1p(d["koi_period"].clip(lower=0))
    d["log_depth"]  = np.log1p(d["koi_depth"].clip(lower=0))
    d["log_snr"]    = np.log1p(d["koi_model_snr"].clip(lower=0))

    # M/R^3 is proportional to mean stellar density
    d["stellar_density_proxy"] = d["koi_smass"] / (d["koi_srad"] ** 3 + eps)

    # impact param / radius ratio - values close to 1 = grazing transit = likely FP
    d["impact_ror_ratio"] = d["koi_impact"] / (d["koi_ror"] + eps)

    # compare observed duration to what Kepler geometry predicts
    # big deviations suggest eccentricity or blended background binaries
    expected = (
        d["koi_period"] * 24.0 / np.pi
        * d["koi_ror"] / (d["koi_dor"] + eps)
        * np.sqrt((1 - d["koi_impact"] ** 2).clip(lower=0))
    )
    d["expected_duration_ratio"] = d["koi_duration"] / (expected + eps)

    return d


print("loading data...")
raw = pd.read_csv(DATA_PATH)
print(f"  shape: {raw.shape}")

available = [c for c in RAW_FEATURES if c in raw.columns]
skipped = [c for c in RAW_FEATURES if c not in raw.columns]
if skipped:
    print(f"  skipped (not in csv): {skipped}")

df = raw[available + [TARGET]].dropna(subset=[TARGET]).copy()
df = add_features(df)

ALL_FEATURES = available + ENGINEERED
print(f"  features: {len(available)} raw + {len(ENGINEERED)} engineered = {len(ALL_FEATURES)} total")

le = LabelEncoder()
y = le.fit_transform(df[TARGET])
X = df[ALL_FEATURES].copy()

print(f"  classes: {list(le.classes_)}")
print(f"  counts: {dict(zip(le.classes_, np.bincount(y).tolist()))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

n_classes = len(le.classes_)

xgb = BalancedXGBClassifier(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)

# HGB handles NaN natively and respects class_weight, good complement to XGBoost
hgb = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.04,
    max_depth=8,
    min_samples_leaf=20,
    l2_regularization=0.1,
    class_weight="balanced",
    random_state=SEED,
)

# RF needs an imputer since it can't handle NaN directly
rf_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )),
])

model = VotingClassifier(
    estimators=[("xgb", xgb), ("hgb", hgb), ("rf", rf_pipe)],
    voting="soft",
    n_jobs=1,
)

print("\ntraining ensemble...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted")
macro_recall = recall_score(y_test, y_pred, average="macro")
macro_prec = precision_score(y_test, y_pred, average="macro")
cm = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

per_class = {}
for cls in le.classes_:
    per_class[cls] = {
        "precision": round(report[cls]["precision"], 3),
        "recall":    round(report[cls]["recall"], 3),
        "f1":        round(report[cls]["f1-score"], 3),
        "support":   int(report[cls]["support"]),
    }

print(f"\n  accuracy:    {acc:.4f}")
print(f"  macro f1:    {macro_f1:.4f}")
print(f"  weighted f1: {weighted_f1:.4f}")
for cls, m in per_class.items():
    print(f"  {cls}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")

print("\nrunning 5-fold CV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
print(f"  cv macro f1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# SHAP on the XGBoost sub-model
feat_imp = []
try:
    print("\ncomputing SHAP values...")
    bg = X_train.sample(min(500, len(X_train)), random_state=SEED)
    explainer = shap.TreeExplainer(model.estimators_[0])
    sv = np.array(explainer.shap_values(bg))

    if sv.ndim == 3:
        mean_shap = np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 2:
        mean_shap = np.abs(sv).mean(axis=0)
    else:
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)

    feat_imp = sorted(zip(ALL_FEATURES, mean_shap.tolist()), key=lambda x: x[1], reverse=True)
    print("  top 5:", [f[0] for f in feat_imp[:5]])

except Exception as e:
    print(f"  SHAP failed: {e}, falling back to RF importances")

if not feat_imp:
    rf_importances = model.estimators_[2].named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(ALL_FEATURES, rf_importances.tolist()), key=lambda x: x[1], reverse=True)

# save everything
joblib.dump({
    "pipeline": model,
    "label_encoder": le,
    "features": ALL_FEATURES,
    "engineered": ENGINEERED,
}, ROOT / "model.joblib")
print(f"\nsaved model ({(ROOT / 'model.joblib').stat().st_size / 1e6:.1f} MB)")

stats = {
    "accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4),
    "weighted_f1": round(weighted_f1, 4),
    "macro_recall": round(macro_recall, 4),
    "macro_precision": round(macro_prec, 4),
    "cv_macro_f1_mean": round(float(cv_scores.mean()), 4),
    "cv_macro_f1_std": round(float(cv_scores.std()), 4),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "n_total": int(len(X)),
    "classes": list(le.classes_),
    "class_counts": {k: int(v) for k, v in zip(le.classes_, np.bincount(y).tolist())},
    "confusion_matrix": cm,
    "per_class": per_class,
    "feature_importance": feat_imp[:15],
    "features": ALL_FEATURES,
    "n_features_raw": len(available),
    "n_features_engineered": len(ENGINEERED),
}
(DATA_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
print("saved stats.json")

# stripped CSV for the notebook (raw features only, engineered ones get recomputed)
df[available + [TARGET]].to_csv(DATA_DIR / "koi_stripped.csv", index=False)
print("done")
