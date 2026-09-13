"""
explainability.py
-------------------
Two complementary, always-on explanation mechanisms for every Zynex
prediction (never a black box, per the PS requirement):

  1. attention_attribution() - reads the model's own temporal + feature
     attention weights (returned directly by WorldModel.forward) to report
     which past windows and which raw features drove the prediction.

  2. shap_attribution() - an optional, model-agnostic cross-check using SHAP
     KernelExplainer against the baseline Logistic Regression's feature space
     (same feature vector the world model consumes at its most recent
     timestep). Requires the `shap` package; degrades gracefully (returns
     None) if unavailable so the rest of the pipeline is never blocked by it.
"""
from __future__ import annotations
import numpy as np


def attention_attribution(temporal_weights: np.ndarray, feature_weights: np.ndarray,
                           feature_names, top_k: int = 5):
    """
    temporal_weights : (T,)     attention over past windows for one sample
    feature_weights  : (T, F)   per-timestep feature attention for one sample
    Returns a dict with the most influential past timestep(s) and the top_k
    most influential features (averaged, temporal-weighted).
    """
    temporal_weights = np.asarray(temporal_weights).reshape(-1)
    feature_weights = np.asarray(feature_weights)

    # feature importance = feature attention weighted by how much that timestep
    # itself mattered (temporal attention) -> single importance score per feature
    weighted_feat = (feature_weights * temporal_weights[:, None]).sum(axis=0)
    order = np.argsort(-weighted_feat)[:top_k]

    top_features = [
        {"feature": str(feature_names[i]), "importance": float(weighted_feat[i])}
        for i in order
    ]
    most_influential_timestep = int(np.argmax(temporal_weights))

    return {
        "top_features": top_features,
        "most_influential_timestep_offset": most_influential_timestep - len(temporal_weights),
        "temporal_weights": temporal_weights.tolist(),
    }


def shap_attribution(baseline_clf, baseline_scaler, background_X: np.ndarray,
                      sample_X: np.ndarray, feature_names, top_k: int = 5):
    """
    Computes SHAP values for a single sample against the baseline classifier's
    decision function, as a model-agnostic cross-check for the attention-based
    explanation above. Returns None if `shap` is not installed.
    """
    try:
        import shap
    except ImportError:
        return None

    background_s = baseline_scaler.transform(background_X)
    sample_s = baseline_scaler.transform(sample_X.reshape(1, -1))

    explainer = shap.KernelExplainer(baseline_clf.predict_proba, background_s[:50])
    shap_values = explainer.shap_values(sample_s, nsamples=100)

    # shap_values[1] = contribution toward the "infiltration" class
    values = np.array(shap_values[1]).reshape(-1)
    order = np.argsort(-np.abs(values))[:top_k]
    return [
        {"feature": str(feature_names[i]), "shap_value": float(values[i])}
        for i in order
    ]


def combine_explanations(attn_result: dict, shap_result):
    """Merges the two explanation sources into one ranked, judge-friendly summary."""
    combined = {"attention": attn_result, "shap": shap_result}
    if shap_result is not None:
        attn_feats = {f["feature"] for f in attn_result["top_features"]}
        shap_feats = {f["feature"] for f in shap_result}
        combined["agreement"] = sorted(attn_feats & shap_feats)
    return combined
