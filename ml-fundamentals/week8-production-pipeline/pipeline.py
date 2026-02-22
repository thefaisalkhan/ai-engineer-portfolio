"""
Production ML Pipeline: sklearn Pipeline → FastAPI → Docker-ready
Pattern used to serve models in production: preprocessing baked into the model artifact.
"""

import joblib
import logging
import numpy as np
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("model.joblib")


# ─── 1. Data ──────────────────────────────────────────────────────────────────

def get_data():
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=12,
        n_redundant=4, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ─── 2. Build Pipeline ────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    sklearn Pipeline bundles preprocessing + model.
    Advantage: can call pipeline.predict(raw_data) in production without
    manually reapplying scalers — prevents train/serve skew.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])


# ─── 3. Train + Evaluate ──────────────────────────────────────────────────────

def train_and_evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    logger.info("Training pipeline...")
    pipeline.fit(X_train, y_train)

    # Cross-validation (5-fold on training set)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Test evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"Test AUC-ROC: {auc:.4f}")
    print(f"CV  AUC-ROC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {"auc": auc, "cv_auc_mean": cv_scores.mean(), "cv_auc_std": cv_scores.std()}


# ─── 4. Save / Load ──────────────────────────────────────────────────────────

def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH):
    joblib.dump(pipeline, path)
    size_kb = path.stat().st_size / 1024
    logger.info(f"Model saved to {path} ({size_kb:.1f} KB)")


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline


# ─── 5. Serve Predictions ────────────────────────────────────────────────────

def predict(pipeline: Pipeline, features: list[list[float]]) -> list[dict]:
    """
    Called by FastAPI endpoint. Returns label + confidence.
    Pipeline handles scaling internally — raw features go in.
    """
    X = np.array(features)
    labels = pipeline.predict(X)
    probas = pipeline.predict_proba(X)

    return [
        {
            "label": int(label),
            "confidence": round(float(probas[i].max()), 4),
            "proba_class_1": round(float(probas[i][1]), 4),
        }
        for i, label in enumerate(labels)
    ]


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data()

    pipeline = build_pipeline()
    metrics = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    save_pipeline(pipeline)

    # Verify load + predict works end-to-end
    loaded = load_pipeline()
    sample = X_test[:3].tolist()
    preds = predict(loaded, sample)
    print("\n=== Sample Predictions ===")
    for i, p in enumerate(preds):
        print(f"  Sample {i}: label={p['label']}, confidence={p['confidence']}, p(class=1)={p['proba_class_1']}")
