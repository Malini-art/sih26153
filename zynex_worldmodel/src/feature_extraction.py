"""
feature_extraction.py
----------------------
Ingests either:
  (a) the Zynex synthetic schema (from synthetic_data.py), or
  (b) a real CIC-IDS-2018 / CTU-13 CSV (auto-detected column schema),
and produces:
  1. Per-time-window aggregated STATE VECTORS combining flow-level and
     packet-level features (the "S_t" of the world model).
  2. Sliding-window SEQUENCES of state vectors of length SEQ_LEN, each paired
     with:
        - the true next-state vector (self-supervised transition target)
        - a binary infiltration label for "does malicious activity occur in
          the next K windows after this sequence" (forecast target)
        - the MITRE ATT&CK stage index at that future horizon
  3. A saved feature scaler (mean/std) for consistent normalisation at
     inference time.

Output: a single .npz file consumed by train.py / evaluate.py / predict_engine.py.

Usage:
    python src/feature_extraction.py --input data/synthetic_traffic.csv --out data/features.npz
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from mitre_mapping import label_from_dataset_stage, STAGE_TO_IDX

WINDOW_SECONDS = 1.0
SEQ_LEN = 20        # number of past windows the world model conditions on
FORECAST_K = 5       # "does infiltration occur within the next K windows"

FEATURE_NAMES = [
    # --- flow-level (aggregated per window) ---
    "n_flows", "syn_ratio", "ack_ratio", "fin_ratio", "rst_ratio", "psh_ratio", "urg_ratio",
    "bytes_mean", "bytes_std", "packets_mean", "duration_mean",
    "iat_mean_mean", "iat_var_mean", "iat_max_mean", "bidir_ratio_mean",
    "unique_dst_ports", "unique_src_ips",
    # --- packet-level (aggregated per window) ---
    "ttl_mean", "ttl_var_mean", "win_size_mean", "frag_ratio",
    "payload_mean", "payload_skew_mean", "scan_score_mean", "retrans_mean",
]


# ----------------------------------------------------------------------------
# Dataset schema adapters
# ----------------------------------------------------------------------------
def _load_synthetic(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _safe_col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    """Returns df[name] as a numeric Series if it exists, else a Series of
    `default` aligned to df's index. Using this everywhere (instead of
    df.get(name, default) followed by .fillna()) avoids the AttributeError
    that happens when df.get() falls back to a bare scalar default -- a bare
    int/float has no .fillna(), only a Series does.
    Also replaces +/-inf with `default`: real CIC-IDS-2017/2018 CSVs contain
    literal "Infinity" strings (e.g. in Flow Bytes/s when duration is 0),
    which pd.to_numeric turns into inf -- NOT NaN -- so a plain .fillna()
    would silently let inf values through into the model."""
    if name in df.columns:
        s = pd.to_numeric(df[name], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan)
        return s.fillna(default)
    return pd.Series(default, index=df.index)

def _load_cic_ids(path: str) -> pd.DataFrame:
    """
    Adapter for CIC-IDS-2017/2018 CSV exports (CICFlowMeter columns).
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    colmap = {
        "Timestamp": "timestamp", "Flow Duration": "duration",
        "Total Fwd Packets": "packets_fwd", "Tot Fwd Pkts": "packets_fwd",
        "Total Backward Packets": "packets_bwd", "Tot Bwd Pkts": "packets_bwd",
        "Total Length of Fwd Packets": "bytes_fwd", "TotLen Fwd Pkts": "bytes_fwd",
        "Total Length of Bwd Packets": "bytes_bwd", "TotLen Bwd Pkts": "bytes_bwd",
        "Flow IAT Mean": "iat_mean", "Flow IAT Std": "iat_var", "Flow IAT Max": "iat_max",
        "SYN Flag Count": "syn", "SYN Flag Cnt": "syn",
        "ACK Flag Count": "ack", "ACK Flag Cnt": "ack",
        "FIN Flag Count": "fin", "FIN Flag Cnt": "fin",
        "RST Flag Count": "rst", "RST Flag Cnt": "rst",
        "PSH Flag Count": "psh", "PSH Flag Cnt": "psh",
        "URG Flag Count": "urg", "URG Flag Cnt": "urg",
        "Destination Port": "dst_port", "Dst Port": "dst_port",
        "Source Port": "src_port", "Src Port": "src_port",
        "Source IP": "src_ip", "Src IP": "src_ip",
        "Destination IP": "dst_ip", "Dst IP": "dst_ip",
        "Init_Win_bytes_forward": "win_size", "Init Fwd Win Byts": "win_size",
        "Label": "label",
    }
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df.columns})

    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df), dtype=float) * 0.01
    else:
        dt = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        if dt.isna().mean() > 0.5:
            df["timestamp"] = np.arange(len(df), dtype=float) * 0.01
        else:
            ts = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
            df["timestamp"] = ts.ffill().bfill()

    bytes_fwd = _safe_col(df, "bytes_fwd")
    bytes_bwd = _safe_col(df, "bytes_bwd")
    packets_fwd = _safe_col(df, "packets_fwd")
    packets_bwd = _safe_col(df, "packets_bwd")

    df["bytes"] = bytes_fwd + bytes_bwd
    df["packets"] = packets_fwd + packets_bwd
    df["duration"] = _safe_col(df, "duration") / 1e6
    df["bidir_ratio"] = packets_bwd / (packets_fwd + packets_bwd + 1e-6)

    for c in ["syn", "ack", "fin", "rst", "psh", "urg", "win_size", "iat_mean", "iat_var", "iat_max"]:
        df[c] = _safe_col(df, c)

    df["dst_port"] = df["dst_port"] if "dst_port" in df.columns else 0
    df["src_port"] = df["src_port"] if "src_port" in df.columns else 0
    df["src_ip"] = df["src_ip"].astype(str) if "src_ip" in df.columns else "unknown"
    df["dst_ip"] = df["dst_ip"].astype(str) if "dst_ip" in df.columns else "unknown"

    df["ttl_mean"] = 64.0
    df["ttl_var"] = 0.5
    df["frag_flag"] = 0
    df["payload_mean"] = df["bytes"] / df["packets"].replace(0, 1)
    df["payload_skew"] = 0.0
    df["scan_score"] = (df["dst_port"].astype(str).apply(lambda x: 1.0 if x not in
                        ("80", "443", "53", "22") else 0.0))
    df["retrans"] = 0

    df["label"] = (df["label"].astype(str) if "label" in df.columns else "Benign")
    df["stage"] = df["label"]
    return df


LOADERS = {
    "synthetic": _load_synthetic,
    "cic-ids-2018": _load_cic_ids,
    "cic-ids-2017": _load_cic_ids,
    "ctu-13": _load_cic_ids,
}

def build_state_vectors(df: pd.DataFrame, window_seconds: float = WINDOW_SECONDS):
    df = df.copy()
    df["window_id"] = (df["timestamp"] // window_seconds).astype(int)

    rows = []
    stage_labels = []
    window_ids = sorted(df["window_id"].unique())

    for wid in window_ids:
        w = df[df["window_id"] == wid]
        n = len(w)
        if n == 0:
            continue
        feat = {
            "n_flows": n,
            "syn_ratio": w["syn"].mean(),
            "ack_ratio": w["ack"].mean(),
            "fin_ratio": w["fin"].mean(),
            "rst_ratio": w["rst"].mean(),
            "psh_ratio": w["psh"].mean(),
            "urg_ratio": w["urg"].mean(),
            "bytes_mean": w["bytes"].mean(),
            "bytes_std": w["bytes"].std(ddof=0) if n > 1 else 0.0,
            "packets_mean": w["packets"].mean(),
            "duration_mean": w["duration"].mean(),
            "iat_mean_mean": w["iat_mean"].mean(),
            "iat_var_mean": w["iat_var"].mean(),
            "iat_max_mean": w["iat_max"].mean(),
            "bidir_ratio_mean": w["bidir_ratio"].mean(),
            "unique_dst_ports": w["dst_port"].nunique(),
            "unique_src_ips": w["src_ip"].nunique(),
            "ttl_mean": w["ttl_mean"].mean(),
            "ttl_var_mean": w["ttl_var"].mean(),
            "win_size_mean": w["win_size"].mean(),
            "frag_ratio": w["frag_flag"].mean(),
            "payload_mean": w["payload_mean"].mean(),
            "payload_skew_mean": w["payload_skew"].mean(),
            "scan_score_mean": w["scan_score"].mean(),
            "retrans_mean": w["retrans"].mean(),
        }
        rows.append([feat[k] for k in FEATURE_NAMES])

        malicious = w[w["label"].astype(str).str.lower() != "benign"]
        if len(malicious) == 0:
            stage_labels.append(STAGE_TO_IDX["Benign"])
        else:
            # majority non-benign stage in this window
            stage_counts = malicious["stage"].apply(label_from_dataset_stage).value_counts()
            stage_labels.append(int(stage_counts.idxmax()))

    X = np.nan_to_num(np.array(rows, dtype=np.float32))
    stage_labels = np.array(stage_labels, dtype=np.int64)
    return X, stage_labels


def build_sequences(X: np.ndarray, stage_labels: np.ndarray, seq_len: int = SEQ_LEN, k: int = FORECAST_K):
    """
    From the per-window state matrix X (T, F), builds:
      seq_X          : (N, seq_len, F)          -- input windows
      next_state_y   : (N, F)                   -- true S_{t+1} (transition target)
      infil_y        : (N,)                     -- 1 if ANY malicious window occurs in (t, t+k]
      stage_y        : (N,)                     -- stage at the first malicious window in that horizon
                                                    (Benign index if none)
    """
    T, F = X.shape
    seq_X, next_state_y, infil_y, stage_y = [], [], [], []

    for t in range(seq_len, T - 1):
        window_in = X[t - seq_len:t]           # past seq_len windows
        target_next = X[t]                      # immediate next state (self-supervised head)

        horizon_end = min(T, t + k)
        horizon_stages = stage_labels[t:horizon_end]
        is_infil = int(np.any(horizon_stages != STAGE_TO_IDX["Benign"]))
        if is_infil:
            nz = horizon_stages[horizon_stages != STAGE_TO_IDX["Benign"]]
            stage = int(nz[0])
        else:
            stage = STAGE_TO_IDX["Benign"]

        seq_X.append(window_in)
        next_state_y.append(target_next)
        infil_y.append(is_infil)
        stage_y.append(stage)

    return (np.array(seq_X, dtype=np.float32),
            np.array(next_state_y, dtype=np.float32),
            np.array(infil_y, dtype=np.int64),
            np.array(stage_y, dtype=np.int64))


def normalise(seq_X, next_state_y, mean=None, std=None):
    if mean is None:
        mean = seq_X.reshape(-1, seq_X.shape[-1]).mean(axis=0)
        std = seq_X.reshape(-1, seq_X.shape[-1]).std(axis=0) + 1e-6
    seq_X_n = (seq_X - mean) / std
    next_state_y_n = (next_state_y - mean) / std
    return seq_X_n, next_state_y_n, mean, std


def main():
    ap = argparse.ArgumentParser(description="Zynex feature extraction pipeline")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--dataset", type=str, default="synthetic", choices=list(LOADERS.keys()))
    ap.add_argument("--out", type=str, default="data/features.npz")
    ap.add_argument("--window_seconds", type=float, default=WINDOW_SECONDS)
    ap.add_argument("--seq_len", type=int, default=SEQ_LEN)
    ap.add_argument("--k", type=int, default=FORECAST_K)
    args = ap.parse_args()

    df = LOADERS[args.dataset](args.input)
    X, stage_labels = build_state_vectors(df, window_seconds=args.window_seconds)
    print(f"[feature_extraction] {X.shape[0]} time windows, {X.shape[1]} features/window")

    seq_X, next_state_y, infil_y, stage_y = build_sequences(X, stage_labels, seq_len=args.seq_len, k=args.k)
    seq_X_n, next_state_y_n, mean, std = normalise(seq_X, next_state_y)

    print(f"[feature_extraction] {seq_X.shape[0]} training sequences "
          f"(infiltration positive rate = {infil_y.mean():.3f})")

    np.savez_compressed(
        args.out,
        seq_X=seq_X_n, next_state_y=next_state_y_n, infil_y=infil_y, stage_y=stage_y,
        mean=mean, std=std, feature_names=np.array(FEATURE_NAMES),
        seq_len=args.seq_len, k=args.k,
    )
    print(f"[feature_extraction] saved -> {args.out}")


if __name__ == "__main__":
    main()
