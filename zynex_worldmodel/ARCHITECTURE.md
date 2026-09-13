# Zynex — Architecture Document
**SIH 2026 | PS #26153 — AI based Network Attack Forecasting from Network Traffic Data**

## 1. Problem Framing
Traditional NIDS classifiers map a single flow to {benign, malicious}, discarding the temporal and
causal structure of an infiltration (recon → foothold → lateral movement → C2 → exfiltration).
Zynex instead learns a **World Model**: an internal simulation of how network state evolves,
`P(S_t+1 | S_t)`, and uses it to forecast whether the *trajectory* the network is currently on
converges toward compromise — before the attacker finishes the kill chain.

## 2. High-Level Pipeline

```
 Raw Telemetry              Feature Layer                 World Model Core            Decision Layer
┌─────────────┐      ┌───────────────────────┐      ┌───────────────────────┐   ┌───────────────────────┐
│ NetFlow/     │      │ Flow-level features   │      │ State encoder         │   │ K-step forward         │
│ IPFIX CSV    │─────▶│  (ports, flags, IAT,  │─────▶│ (dense embedding)     │──▶│ simulation             │
│ PCAP (Scapy/ │      │   bytes, duration)    │      │        ↓              │   │  → infiltration prob.  │
│ PyShark)     │      │ Packet-level features │      │ LSTM / Temporal       │   │  → MITRE ATT&CK stage  │
└─────────────┘      │  (TTL var, window sz,  │      │ Self-Attention over   │   │  → driving features    │
                      │   frag flags, retrans) │      │ sliding windows       │   │    (attention/SHAP)    │
                      └───────────────────────┘      │        ↓              │   └───────────────────────┘
                                                      │ Dual heads:           │
                                                      │  (a) next-state       │
                                                      │      regression       │
                                                      │  (b) infiltration &   │
                                                      │      stage classifier │
                                                      └───────────────────────┘
```

## 3. State Representation
At each time window `t` (default 2s, configurable), Zynex aggregates all flows into a fixed-length
**state vector** `S_t`:

- **Flow-level (12 features):** src/dst port entropy, TCP flag distribution (SYN/ACK/FIN/RST/PSH/URG
  ratios), protocol mix, bytes/packets per flow, flow duration stats, inter-arrival-time (mean/var/max),
  bidirectional flow ratio.
- **Packet-level (8 features):** TTL mean/variance, TCP window size mean, fragment-flag ratio, payload
  size distribution (mean/skew), sequential vs randomised port-access score (scan signature), retransmission
  count.
- A `GraphOption` mode additionally builds a host-communication graph per window (nodes = IPs, edges =
  flow volume) and pools node embeddings via a lightweight message-passing layer — enabled with
  `--encoder gnn` in `train.py` — to capture lateral-movement topology, satisfying the "or Graph Neural
  Networks" option in the PS.

Sequences of `S_t` (default window = 20 timesteps) form the training examples for the sequence model.

## 4. World Model Core
- **Encoder:** 2-layer LSTM (hidden=128) over the windowed state sequence, wrapped with a
  **temporal self-attention** layer (single head, scaled dot-product) that produces per-timestep and
  per-feature attention weights — used directly for explainability, not just as a black box.
- **Head A — Transition Dynamics (self-supervised):** predicts `Ŝ_t+1` from `S_1..S_t`; trained with
  MSE against the true next state. This is what makes the model a *world model* rather than a
  classifier — it is optimised to understand how the network evolves, and the infiltration head is
  layered on top of the same learned representation.
- **Head B — Infiltration Forecast (supervised):** a small MLP on the LSTM's final hidden state outputs
  (i) `P(infiltration within next K windows)` and (ii) a softmax over 6 classes
  `{Benign, Reconnaissance, Initial Access, Lateral Movement, Command & Control, Exfiltration}`,
  trained against ground-truth labels derived from the dataset's attack timeline.
- **K-step rollout:** at inference, the model repeatedly feeds its own predicted `Ŝ_t+1` back in as
  input (closed-loop simulation) for `K` steps, re-scoring infiltration probability and MITRE stage at
  each simulated step, producing a **probability-over-time curve** rather than a single number.

## 5. MITRE ATT&CK Mapping (`src/mitre_mapping.py`)
Predicted stage probabilities are combined with rule-based corroboration (e.g., sequential port sweep →
Reconnaissance prior; SYN-flood signature + repeated auth failures → Initial Access/Brute-Force; sudden
increase in internal east-west flows → Lateral Movement; long-lived low-volume periodic flows to a single
external IP → Command & Control; large sustained outbound transfer → Exfiltration), giving a hybrid
learned + heuristic stage label the demo can explain in plain language.

## 6. Explainability (`src/explainability.py`)
Two complementary, always-on explanation mechanisms (never a black box):
1. **Attention attribution:** the temporal self-attention weights directly show which past timesteps
   and which input features (via a feature-gating attention row) most influenced the current prediction.
2. **SHAP (KernelExplainer, optional):** for the classifier head, SHAP values are computed against the
   baseline logistic-regression feature space for a model-agnostic, judge-friendly attribution that can
   be cross-checked against the attention output.
Both are surfaced together in the Streamlit demo as a ranked "top contributing features" panel.

## 7. Baseline & Benchmarking
`src/baseline_model.py` trains a Logistic Regression on the same flattened feature windows with no
temporal context (single-window classification). `src/evaluate.py` reports F1, Precision, Recall and
False-Positive-Rate for both models on an identical held-out split — quantifying the uplift gained from
modelling transition dynamics instead of static classification.

## 8. Deployment
Everything (feature pipeline, trained model, inference engine, and Streamlit UI) runs **fully offline**
on a single machine — no cloud API dependency — making it deployable inside air-gapped enterprise or
Critical Information Infrastructure (CII) SOC environments. Model weights and configs are versioned
under `models/` for reproducibility.
