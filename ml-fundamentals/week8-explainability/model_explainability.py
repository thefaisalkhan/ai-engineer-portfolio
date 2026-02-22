"""
Phase 2 — Week 8: Model Explainability
========================================
Covers: SHAP (TreeExplainer, KernelExplainer), LIME, permutation importance
Job relevance: 74% of AI/ML job postings; required in finance, healthcare, HR AI
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular


def load_and_prepare_data():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ── SHAP: TreeExplainer (for tree-based models) ───────────────────────────────
def shap_tree_explainer(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """
    TreeExplainer: fast, exact SHAP values for tree-based models.
    SHAP values = additive feature contributions that sum to the prediction.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values is a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 (malignant)
    else:
        shap_vals = shap_values

    mean_abs_shap = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    # Local explanation for a single prediction
    sample_idx = 0
    local_explanation = pd.DataFrame({
        "feature": X_test.columns,
        "feature_value": X_test.iloc[sample_idx].values,
        "shap_value": shap_vals[sample_idx],
    }).sort_values("shap_value", key=abs, ascending=False)

    return {
        "global_importance": mean_abs_shap,
        "local_explanation_sample_0": local_explanation,
        "expected_value": float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value),
        "prediction_sample_0": float(model.predict_proba(X_test.iloc[[0]])[0][1]),
    }


# ── SHAP: KernelExplainer (model-agnostic) ────────────────────────────────────
def shap_kernel_explainer(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_background: int = 50,
    n_explain: int = 10,
) -> dict:
    """
    KernelExplainer: works on ANY model (logistic regression, neural networks, etc.)
    Slower than TreeExplainer — use a background dataset summary.
    """
    background = shap.sample(X_train, n_background, random_state=42)
    explainer = shap.KernelExplainer(
        model.predict_proba, background, link="logit"
    )
    shap_values = explainer.shap_values(X_test.iloc[:n_explain], nsamples=100)

    shap_vals = shap_values[1]  # class 1
    mean_abs_shap = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "global_importance_top5": mean_abs_shap.head(5),
        "n_samples_explained": n_explain,
        "method": "SHAP KernelExplainer (model-agnostic)",
    }


# ── LIME: Local Interpretable Model-Agnostic Explanations ────────────────────
def lime_explanation(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    sample_idx: int = 0,
) -> dict:
    """
    LIME fits a local linear model around a single prediction.
    Useful for explaining individual predictions to non-technical stakeholders.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["benign", "malignant"],
        mode="classification",
        random_state=42,
    )

    explanation = explainer.explain_instance(
        data_row=X_test.iloc[sample_idx].values,
        predict_fn=model.predict_proba,
        num_features=10,
    )

    lime_features = pd.DataFrame(
        explanation.as_list(),
        columns=["feature_condition", "lime_weight"],
    ).sort_values("lime_weight", key=abs, ascending=False)

    return {
        "sample_index": sample_idx,
        "predicted_class": "malignant" if model.predict(X_test.iloc[[sample_idx]])[0] == 1 else "benign",
        "predicted_probability": float(model.predict_proba(X_test.iloc[[sample_idx]])[0][1]),
        "lime_features": lime_features,
        "interpretation": "Positive weight → pushes toward malignant; negative → pushes toward benign",
    }


# ── Permutation Feature Importance ──────────────────────────────────────────
def permutation_feature_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """
    Gold-standard feature importance: permute feature, measure accuracy drop.
    Unlike built-in importance, unbiased for high-cardinality features.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


# ── Compare explainability across 3 models ───────────────────────────────────
def compare_model_explanations(X_train, X_test, y_train, y_test) -> dict:
    """Compare SHAP importance from 3 different model types."""
    models = {
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        top5 = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=X_test.columns,
        ).nlargest(5)
        results[name] = top5

    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Train a GBM
    gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)
    acc = gbm.score(X_test, y_test)
    print(f"GBM Test Accuracy: {acc:.4f}\n")

    print("=== SHAP TreeExplainer ===")
    shap_result = shap_tree_explainer(gbm, X_train, X_test)
    print("Top 10 features by mean |SHAP|:")
    print(shap_result["global_importance"].head(10).to_string(index=False))
    print(f"\nExpected value (base rate): {shap_result['expected_value']:.4f}")
    print(f"Sample 0 prediction:        {shap_result['prediction_sample_0']:.4f}")

    print("\n=== LIME Explanation (sample 0) ===")
    lime_result = lime_explanation(gbm, X_train, X_test, sample_idx=0)
    print(f"Predicted: {lime_result['predicted_class']} "
          f"(p={lime_result['predicted_probability']:.4f})")
    print(lime_result["lime_features"].to_string(index=False))

    print("\n=== Permutation Feature Importance ===")
    perm = permutation_feature_importance(gbm, X_test, y_test)
    print(perm.head(10).to_string(index=False))

    print("\n=== Cross-Model SHAP Comparison ===")
    comparisons = compare_model_explanations(X_train, X_test, y_train, y_test)
    for model_name, importance in comparisons.items():
        print(f"\n{model_name} — top 5 features:")
        for feat, val in importance.items():
            print(f"  {feat}: {val:.4f}")
