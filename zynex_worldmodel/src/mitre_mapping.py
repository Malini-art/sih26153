"""
mitre_mapping.py
-----------------
Maps predicted / observed network-state features to MITRE ATT&CK tactics.
Used two ways in Zynex:
 1. To derive GROUND-TRUTH stage labels from a dataset's attack-timeline
    annotations at training time (label_from_dataset_stage).
 2. To provide a rule-based CORROBORATION prior that is blended with the
    world model's learned stage-classification head at inference time
    (heuristic_stage_scores), so the demo can show *why* a stage was
    predicted in plain language, not just a softmax number.
"""
from __future__ import annotations
import numpy as np

STAGES = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement",
          "Command & Control", "Exfiltration"]

STAGE_TO_IDX = {s: i for i, s in enumerate(STAGES)}
IDX_TO_STAGE = {i: s for i, s in enumerate(STAGES)}

# Human-readable rationale templates surfaced in the demo UI.
STAGE_RATIONALE = {
    "Reconnaissance": "Sequential/randomised port-sweep signature against a single host with low "
                       "byte counts and very short inter-arrival times.",
    "Initial Access": "Repeated authentication/service-exploit attempts (SSH/RDP/SMB/FTP ports) with "
                       "elevated SYN/RST ratio and short flow durations.",
    "Lateral Movement": "Sharp increase in internal east-west (host-to-host) flows using SMB/RPC/RDP "
                         "ports with moderate-to-high byte volume.",
    "Command & Control": "Low-volume, highly periodic beaconing to a single external IP with unusually "
                          "stable inter-arrival time (low variance) — classic C2 heartbeat pattern.",
    "Exfiltration": "Large, sustained outbound byte volume to an external IP with high packet count and "
                     "long flow duration.",
    "Benign": "Traffic pattern consistent with normal user/application behaviour.",
}


def label_from_dataset_stage(stage_str: str) -> int:
    """Maps a dataset's raw stage/label string onto the canonical Zynex stage index."""
    if stage_str is None:
        return STAGE_TO_IDX["Benign"]
    s = str(stage_str).strip().lower()
    if s in ("benign", "normal", "0", ""):
        return STAGE_TO_IDX["Benign"]
    mapping = {
        "reconnaissance": "Reconnaissance", "recon": "Reconnaissance", "portscan": "Reconnaissance",
        "port scan": "Reconnaissance", "probe": "Reconnaissance",
        "initial access": "Initial Access", "bruteforce": "Initial Access", "brute force": "Initial Access",
        "ftp-patator": "Initial Access", "ssh-patator": "Initial Access", "webattack": "Initial Access",
        "lateral movement": "Lateral Movement", "lateral": "Lateral Movement",
        "command & control": "Command & Control", "c2": "Command & Control", "botnet": "Command & Control",
        "exfiltration": "Exfiltration", "exfil": "Exfiltration", "ddos": "Exfiltration", "dos": "Exfiltration",
        "infiltration": "Exfiltration",
    }
    for key, canon in mapping.items():
        if key in s:
            return STAGE_TO_IDX[canon]
    # unknown-but-malicious dataset label -> fall back to the most common early stage
    return STAGE_TO_IDX["Reconnaissance"]


def heuristic_stage_scores(feat_row: dict) -> np.ndarray:
    """
    Rule-based prior over the 6 stages given one aggregated window's feature dict
    (keys match feature_extraction.FEATURE_NAMES). Returns an unnormalised score
    vector aligned with STAGES; caller may softmax/blend with the learned head.
    """
    scores = np.zeros(len(STAGES), dtype=np.float32)

    scan = feat_row.get("scan_score", 0.0)
    syn_ratio = feat_row.get("syn_ratio", 0.0)
    bytes_mean = feat_row.get("bytes_mean", 0.0)
    duration_mean = feat_row.get("duration_mean", 0.0)
    iat_var = feat_row.get("iat_var_mean", 1.0)
    iat_mean = feat_row.get("iat_mean_mean", 0.0)
    internal_ratio = feat_row.get("bidir_ratio_mean", 0.0)
    retrans = feat_row.get("retrans_mean", 0.0)
    packets_mean = feat_row.get("packets_mean", 0.0)

    scores[STAGE_TO_IDX["Reconnaissance"]] = scan * 2.0 + (1.0 if bytes_mean < 100 else 0.0)
    scores[STAGE_TO_IDX["Initial Access"]] = syn_ratio * 1.5 + (1.0 if 100 <= bytes_mean < 400 else 0.0)
    scores[STAGE_TO_IDX["Lateral Movement"]] = internal_ratio * 1.5 + (1.0 if 500 <= bytes_mean < 5000 else 0.0)
    scores[STAGE_TO_IDX["Command & Control"]] = (1.0 if iat_mean > 5.0 and iat_var < 5.0 else 0.0) * 2.0
    scores[STAGE_TO_IDX["Exfiltration"]] = (1.0 if bytes_mean > 20000 or packets_mean > 100 else 0.0) * 2.0 + retrans * 0.2
    scores[STAGE_TO_IDX["Benign"]] = max(0.1, 1.0 - scores.sum())

    return scores


def blend_scores(model_probs: np.ndarray, heuristic: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """alpha weights the learned model's softmax; (1-alpha) weights the rule-based prior."""
    h = heuristic / (heuristic.sum() + 1e-8)
    blended = alpha * model_probs + (1 - alpha) * h
    return blended / blended.sum()
