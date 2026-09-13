"""
evaluate.py
------------
Benchmarks Zynex's World Model against the non-temporal Logistic Regression
baseline on an IDENTICAL held-out test split, reporting F1, Precision, Recall
and False-Positive-Rate for both -- the comparison explicitly required by the
problem statement.

Usage:
    python src/evaluate.py --data data/features.npz --model models/world_model.pt \
        --baseline models/baseline.pkl --out outputs/benchmark_results.json
"""
import argparse
import json
import pickle
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from world_model import WorldModel


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def eval_world_model(model_path, npz_path, test_frac=0.2, device="cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = WorldModel(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
                        n_layers=ckpt["n_layers"], encoder=ckpt.get("encoder", "lstm"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    d = np.load(npz_path, allow_pickle=True)
    seq_X, infil_y = d["seq_X"], d["infil_y"]
    n = len(seq_X)
    split = int(n * (1 - test_frac))
    X_test, y_test = seq_X[split:], infil_y[split:]

    with torch.no_grad():
        out = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.sigmoid(out["infil_logit"]).numpy()
    preds = (probs > 0.5).astype(int)

    return {
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "false_positive_rate": float(false_positive_rate(y_test, preds)),
        "n_test": int(len(y_test)),
    }


def eval_baseline(baseline_path, npz_path, test_frac=0.2):
    with open(baseline_path, "rb") as f:
        saved = pickle.load(f)
    clf, scaler = saved["model"], saved["scaler"]

    d = np.load(npz_path, allow_pickle=True)
    seq_X, infil_y = d["seq_X"], d["infil_y"]
    n = len(seq_X)
    split = int(n * (1 - test_frac))
    X_test_last = seq_X[split:, -1, :]
    y_test = infil_y[split:]

    preds = clf.predict(scaler.transform(X_test_last))

    return {
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "false_positive_rate": float(false_positive_rate(y_test, preds)),
        "n_test": int(len(y_test)),
    }


def main():
    ap = argparse.ArgumentParser(description="Benchmark Zynex World Model vs Logistic Regression baseline")
    ap.add_argument("--data", type=str, default="data/features.npz")
    ap.add_argument("--model", type=str, default="models/world_model.pt")
    ap.add_argument("--baseline", type=str, default="models/baseline.pkl")
    ap.add_argument("--out", type=str, default="outputs/benchmark_results.json")
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    world_model_metrics = eval_world_model(args.model, args.data, args.test_frac)
    baseline_metrics = eval_baseline(args.baseline, args.data, args.test_frac)

    results = {"world_model": world_model_metrics, "baseline_logistic_regression": baseline_metrics}

    print("\n=== Zynex Benchmark: World Model vs. Baseline ===")
    print(f"{'Metric':<22}{'World Model':>15}{'Baseline (LR)':>18}")
    for metric in ["f1", "precision", "recall", "false_positive_rate"]:
        print(f"{metric:<22}{world_model_metrics[metric]:>15.4f}{baseline_metrics[metric]:>18.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[evaluate] saved -> {args.out}")


if __name__ == "__main__":
    main()
