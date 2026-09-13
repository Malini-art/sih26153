"""
predict_engine.py
-------------------
The forward-simulation inference engine required by the PS: given a current
traffic snapshot (the most recent SEQ_LEN windows), performs a closed-loop
K-step forward simulation of the world model, producing:

  - a time-series infiltration probability score for each of the next K
    simulated windows,
  - the predicted MITRE ATT&CK stage at each simulated step (learned head
    blended with the rule-based heuristic prior from mitre_mapping.py),
  - the top contributing traffic features (via explainability.py).

At each simulated step, the model's own predicted next-state Ŝ_t+1 is fed
back in as input for the following step -- true forward simulation, not just
repeated one-step prediction on ground truth.

Usage:
    python src/predict_engine.py --data data/features.npz --model models/world_model.pt --k 5
"""
import argparse
import numpy as np
import torch

from world_model import WorldModel
from mitre_mapping import IDX_TO_STAGE, STAGE_RATIONALE, heuristic_stage_scores, blend_scores
from explainability import attention_attribution


def load_model(model_path, device="cpu"):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = WorldModel(n_features=ckpt["n_features"], hidden_size=ckpt["hidden_size"],
                        n_layers=ckpt["n_layers"], encoder=ckpt.get("encoder", "lstm"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"], ckpt["feature_names"]


def _unnormalise(vec, mean, std):
    return vec * std + mean


def rollout(model, initial_seq: np.ndarray, mean, std, feature_names, k: int = 5, device="cpu"):
    """
    initial_seq : (seq_len, F) NORMALISED state window (most recent traffic snapshot)
    Returns a list of dicts, one per simulated future step.
    """
    seq = torch.tensor(initial_seq, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, F)
    results = []

    for step in range(1, k + 1):
        with torch.no_grad():
            out = model(seq)

        infil_prob = torch.sigmoid(out["infil_logit"]).item()
        stage_probs_model = torch.softmax(out["stage_logits"], dim=-1).squeeze(0).cpu().numpy()

        # explainability for this step
        temporal_w = out["temporal_weights"].squeeze(0).cpu().numpy()
        feature_w = out["feature_weights"].squeeze(0).cpu().numpy()
        expl = attention_attribution(temporal_w, feature_w, feature_names, top_k=5)

        # blend with rule-based heuristic (computed on the *last real* window, unnormalised)
        last_window_raw = _unnormalise(seq.squeeze(0)[-1].cpu().numpy(), mean, std)
        feat_dict = dict(zip(feature_names, last_window_raw))
        heuristic = heuristic_stage_scores(feat_dict)
        blended = blend_scores(stage_probs_model, heuristic, alpha=0.7)
        stage_idx = int(np.argmax(blended))
        stage_name = IDX_TO_STAGE[stage_idx]

        results.append({
            "step": step,
            "infiltration_probability": round(float(infil_prob), 4),
            "predicted_stage": stage_name,
            "stage_confidence": round(float(blended[stage_idx]), 4),
            "rationale": STAGE_RATIONALE.get(stage_name, ""),
            "top_contributing_features": expl["top_features"],
        })

        # closed-loop rollout: predicted next state becomes the newest timestep
        next_state = out["next_state_pred"].detach()          # (1, F)
        seq = torch.cat([seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)

    return results


def main():
    ap = argparse.ArgumentParser(description="Zynex K-step infiltration forward simulation")
    ap.add_argument("--data", type=str, default="data/features.npz")
    ap.add_argument("--model", type=str, default="models/world_model.pt")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sample_index", type=int, default=-1, help="which sequence in the dataset to simulate from (-1 = last)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, feature_names = load_model(args.model, device)

    d = np.load(args.data, allow_pickle=True)
    seq_X = d["seq_X"]
    idx = args.sample_index if args.sample_index >= 0 else len(seq_X) - 1
    initial_seq = seq_X[idx]

    results = rollout(model, initial_seq, mean, std, feature_names, k=args.k, device=device)

    print(f"\n=== Zynex K-step Infiltration Forecast (from window index {idx}) ===")
    for r in results:
        print(f"\n-- t+{r['step']} --")
        print(f"  Infiltration probability : {r['infiltration_probability']:.2%}")
        print(f"  Predicted MITRE stage    : {r['predicted_stage']}  (confidence={r['stage_confidence']:.2f})")
        print(f"  Rationale                : {r['rationale']}")
        print(f"  Top contributing features: " +
              ", ".join(f"{f['feature']} ({f['importance']:.3f})" for f in r["top_contributing_features"]))


if __name__ == "__main__":
    main()
