"""
test_pipeline.py
------------------
Smoke tests covering the full Zynex pipeline. Torch-dependent tests are
skipped automatically if PyTorch is not installed in the current environment,
so `pytest` still gives useful signal on a minimal machine.

Run with:
    pytest tests/test_pipeline.py -v
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from synthetic_data import generate
from feature_extraction import build_state_vectors, build_sequences, normalise, FEATURE_NAMES
from mitre_mapping import label_from_dataset_stage, heuristic_stage_scores, blend_scores, STAGES

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def test_synthetic_data_generation():
    df = generate(n_windows=700, seed=1)
    assert len(df) > 0
    assert set(df["label"].unique()) <= {"Benign", "Malicious"}
    assert (df["label"] == "Malicious").sum() > 0, "campaign should inject malicious flows"


def test_feature_extraction_shapes():
    df = generate(n_windows=800, seed=1)
    X, stage_labels = build_state_vectors(df)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(stage_labels) == X.shape[0]

    seq_X, next_state_y, infil_y, stage_y = build_sequences(X, stage_labels, seq_len=10, k=5)
    assert seq_X.shape[1:] == (10, len(FEATURE_NAMES))
    assert next_state_y.shape[1] == len(FEATURE_NAMES)
    assert set(np.unique(infil_y)).issubset({0, 1})

    seq_X_n, next_state_y_n, mean, std = normalise(seq_X, next_state_y)
    assert seq_X_n.shape == seq_X.shape
    assert mean.shape[0] == len(FEATURE_NAMES)


def test_mitre_mapping():
    assert label_from_dataset_stage("PortScan") == STAGES.index("Reconnaissance")
    assert label_from_dataset_stage("Benign") == STAGES.index("Benign")
    assert label_from_dataset_stage("DDoS") == STAGES.index("Exfiltration")

    feat_row = {"scan_score": 0.9, "bytes_mean": 50, "syn_ratio": 0.1, "duration_mean": 0.01,
                "iat_var_mean": 0.01, "iat_mean_mean": 0.01, "bidir_ratio_mean": 0.1,
                "retrans_mean": 0.0, "packets_mean": 1.0}
    scores = heuristic_stage_scores(feat_row)
    assert scores.shape[0] == len(STAGES)
    assert np.argmax(scores) == STAGES.index("Reconnaissance")

    blended = blend_scores(np.ones(len(STAGES)) / len(STAGES), scores)
    assert np.isclose(blended.sum(), 1.0, atol=1e-4)


def test_baseline_model():
    from baseline_model import load_last_window_features, train_baseline
    df = generate(n_windows=1500, seed=2)
    X, stage_labels = build_state_vectors(df)
    seq_X, next_state_y, infil_y, stage_y = build_sequences(X, stage_labels, seq_len=10, k=5)
    seq_X_n, _, _, _ = normalise(seq_X, next_state_y)

    X_last = seq_X_n[:, -1, :]
    clf, scaler, (X_test, y_test) = train_baseline(X_last, infil_y, test_frac=0.2)
    acc = clf.score(X_test, y_test)
    assert 0.0 <= acc <= 1.0


def test_world_model_forward_pass():
    if not TORCH_AVAILABLE:
        print("SKIP: torch not installed in this environment")
        return
    import torch
    from world_model import WorldModel, world_model_loss

    B, T, F = 4, 20, len(FEATURE_NAMES)
    model = WorldModel(n_features=F)
    x = torch.randn(B, T, F)
    out = model(x)
    assert out["next_state_pred"].shape == (B, F)
    assert out["infil_logit"].shape == (B,)
    assert out["stage_logits"].shape == (B, 6)
    assert out["temporal_weights"].shape == (B, T)

    loss, parts = world_model_loss(
        out, torch.randn(B, F), torch.randint(0, 2, (B,)), torch.randint(0, 6, (B,))
    )
    assert loss.item() > 0


if __name__ == "__main__":
    test_synthetic_data_generation()
    test_feature_extraction_shapes()
    test_mitre_mapping()
    test_baseline_model()
    test_world_model_forward_pass()
    print("All smoke tests passed.")
