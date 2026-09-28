# Zynex — AI-based Network Attack Forecasting from Network Traffic Data
### SIH 2026 | Problem Statement ID: 26153 | Theme: Blockchain & Cybersecurity | Category: Software
### Team: Zynex

Zynex is a **World-Model-based predictive cyber-defence prototype**. Instead of classifying
individual flows as benign/malicious (the traditional static IDS approach), Zynex learns the
**temporal state-transition dynamics** of a network — `P(S_t+1 | S_t)` — and rolls this model
forward in time to estimate the **probability of infiltration in the next K time windows**,
the **predicted MITRE ATT&CK stage**, and **which features are driving that prediction**.

---

## 1. Key Features

| Requirement (from PS) | Implementation |
|---|---|
| Flow + packet-level features | `src/feature_extraction.py` |
| State-transition dynamics via LSTM/Transformer/GNN | `src/world_model.py` (LSTM + temporal self-attention core; GNN-style adjacency mode included) |
| K-step forward simulation | `src/predict_engine.py` |
| MITRE ATT&CK stage mapping | `src/mitre_mapping.py` |
| Explainability (attention / SHAP) | `src/explainability.py` |
| Offline demo interface | `app/streamlit_app.py` |
| Baseline comparison | `src/baseline_model.py` + `src/evaluate.py` |
| Works on CIC-IDS-2018 / CTU-13 | `src/feature_extraction.py --dataset` flag |
| Works without any dataset (judge demo) | `src/synthetic_data.py` generates a realistic labelled traffic stream |

---

## 2. Project Structure

```
zynex_worldmodel/
├── README.md
├── ARCHITECTURE.md          <- 2-page architecture doc (deliverable)
├── requirements.txt
├── data/                    <-  CIC-IDS-2018 / dataset here (or use synthetic data)
├── models/                  <- trained weights land here after train.py
├── outputs/                 <- benchmark_results.json, plots, predictions
├── src/
│   ├── synthetic_data.py    <- offline synthetic CIC-IDS-2018-style generator
│   ├── feature_extraction.py<- flow + packet feature engineering & time-windowing
│   ├── mitre_mapping.py     <- attack-stage heuristics -> MITRE ATT&CK
│   ├── world_model.py       <- LSTM/Transformer world model (PyTorch)
│   ├── baseline_model.py    <- Logistic Regression baseline (scikit-learn)
│   ├── train.py             <- trains the world model
│   ├── predict_engine.py    <- K-step rollout + infiltration scoring
│   ├── explainability.py    <- attention weights + SHAP attribution
│   ├── evaluate.py          <- F1 / precision / recall / FPR benchmark
│   └── utils.py
├── app/
│   └── streamlit_app.py     <- offline demo UI (upload CSV/PCAP-derived CSV)
└── tests/
    └── test_pipeline.py     <- smoke tests for the whole pipeline
```

---

## 3. Setup (Windows + VS Code)

```powershell
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate
cd Zynex_worldmodel
# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Place real datasets in data/
#    - CIC-IDS-2018: https://www.unb.ca/cic/datasets/ids-2018.html or
#    - CTU-13:       https://www.stratosphereips.org/datasets-ctu13
#    If you skip this step, the pipeline auto-generates a synthetic
#    CIC-IDS-2018-style labelled dataset so everything still runs end-to-end.
```

## 4. Running the full pipeline

```powershell
# Step 1 — generate synthetic data (skip if you have a real dataset CSV in data/)
python src\synthetic_data.py --out data\synthetic_traffic.csv --n_windows 4000

# Step 2 — extract flow+packet features and build time-windowed state sequences
python src\feature_extraction.py --input data\synthetic_traffic.csv --out data\feature.npz

# Step 3 — train the world model (LSTM + attention) and the baseline
python src\train.py --data data\features.npz --epochs 15 --out models\world_model.pt
python src\baseline_model.py --data data\features.npz --out models\baseline.pkl

# Step 4 — benchmark world model vs baseline
python src\evaluate.py --data data\features.npz --model models\world_model.pt --baseline models\baseline.pkl --out outputs\benchmark_results.json

# Step 5 — run K-step infiltration prediction + explainability on a sample window
python src\predict_engine.py --data data\features.npz --model models\world_model.pt --k 5

# Step 6 — launch the offline demo UI
streamlit run app\streamlit_app.py
```

Everything above runs **fully offline** — no cloud API calls anywhere in the pipeline.

## 5. Using a real dataset (CIC-IDS-2018)

```powershell
python src\feature_extraction.py --input "data\cicids2018-csv\datasets\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv" --dataset cic-ids-2018 --out data\features_real.npz

python src\train.py --data data\features_real.npz --epochs 15 --out models\world_model_real.pt

python src\baseline_model.py --data data\features_real.npz --out models\baseline_real.pkl

python src\evaluate.py --data data\features_real.npz --model models\world_model_real.pt --baseline models\baseline_real.pkl --out outputs\benchmark_real.json

python src\predict_engine.py --data data\features_real.npz --model models\world_model_real.pt --k 5




streamlit run app\streamlit_app.py 

```
`feature_extraction.py` auto-detects the CIC-IDS-2018  column schema, normalises it into
the common Zynex state-vector format, and derives ground-truth state-transition and MITRE-stage
labels from the dataset's attack timeline annotations (`Label` / `Attack` columns).

## 6. Benchmarking methodology
`evaluate.py` reports F1, Precision, Recall and False-Positive-Rate for:
1. **Zynex World Model** (temporal, sequence-aware)
2. **Logistic Regression baseline** (same features, no temporal context, single-flow classification)

on an identical held-out test split, demonstrating the uplift from modelling transition dynamics.

## 7. Team
**Zynex** — SIH 2026, PS #26153 (National Technical Research Organisation — NTRO)
