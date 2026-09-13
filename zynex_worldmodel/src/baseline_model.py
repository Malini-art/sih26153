"""
baseline_model.py
-------------------
Logistic Regression baseline: classifies infiltration (binary) using ONLY the
flattened features of the single most recent window (t) -- i.e. exactly the
static, temporally-blind approach the PS asks us to demonstrably beat.
No sequence context, no transition modelling.

Usage:
    python src/baseline_model.py --data data/features.npz --out models/baseline.pkl
"""
import argparse
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def load_last_window_features(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    seq_X = d["seq_X"]           # (N, seq_len, F)
    infil_y = d["infil_y"]       # (N,)
    # baseline only sees the CURRENT window's flow features (last timestep), not history
    X_last = seq_X[:, -1, :]
    return X_last, infil_y


def train_baseline(X, y, test_frac=0.2):
    n = len(X)
    split = int(n * (1 - test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    return clf, scaler, (X_test_s, y_test)


def main():
    ap = argparse.ArgumentParser(description="Train Logistic Regression baseline (non-temporal)")
    ap.add_argument("--data", type=str, default="data/features.npz")
    ap.add_argument("--out", type=str, default="models/baseline.pkl")
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    X, y = load_last_window_features(args.data)
    clf, scaler, (X_test, y_test) = train_baseline(X, y, test_frac=args.test_frac)

    acc = clf.score(X_test, y_test)
    print(f"[baseline_model] test accuracy = {acc:.4f}  (n_test={len(y_test)}, positive_rate={y_test.mean():.3f})")

    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "test_frac": args.test_frac}, f)
    print(f"[baseline_model] saved -> {args.out}")


if __name__ == "__main__":
    main()
