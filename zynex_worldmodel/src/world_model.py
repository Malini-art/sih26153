"""
world_model.py
----------------
Zynex's core World Model: learns P(S_t+1 | S_t) over windowed network state
sequences, and is jointly trained with two forecasting heads (infiltration
probability + MITRE ATT&CK stage). Built with PyTorch.

Components:
  - FeatureGateAttention : per-timestep, per-feature attention gate applied to
                            the raw input -- gives "which flags/ports/flow
                            stats mattered" explainability for free.
  - GNNBlock (optional)   : one lightweight message-passing layer treating the
                            feature vector as a fully-connected mini-graph of
                            "signal nodes" -- satisfies the PS's "Graph Neural
                            Networks" option as an alternative encoder.
  - Encoder               : 2-layer LSTM over the (optionally GNN-preprocessed)
                            gated sequence.
  - TemporalAttention     : scaled additive attention pooling over LSTM hidden
                            states -- gives "which past timesteps mattered".
  - Heads                 : (a) next-state regression (self-supervised
                            transition dynamics), (b) infiltration probability,
                            (c) MITRE stage classification.

Run this file directly for a quick shape/sanity self-test:
    python src/world_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGateAttention(nn.Module):
    """Learns a softmax attention gate over input features at every timestep."""

    def __init__(self, n_features: int):
        super().__init__()
        self.gate = nn.Linear(n_features, n_features)

    def forward(self, x):  # x: (B, T, F)
        scores = self.gate(x)
        weights = F.softmax(scores, dim=-1)   # (B, T, F) -- per-feature attention
        return x * weights, weights


class GNNBlock(nn.Module):
    """
    Minimal message-passing layer over a fully-connected "feature graph":
    every feature is treated as a node; one round of message passing lets
    each feature's representation be informed by all others before the
    temporal encoder sees it. This is the GNN alternative referenced in the
    problem statement ("Graph Neural Networks ... or other AI techniques").
    """

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.node_proj = nn.Linear(1, hidden)
        self.message = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (B, T, F)
        B, T, Fdim = x.shape
        nodes = self.node_proj(x.unsqueeze(-1))          # (B, T, F, hidden)
        agg = nodes.mean(dim=2, keepdim=True)              # mean-field message from all nodes
        messages = self.message(agg).expand(-1, -1, Fdim, -1)
        updated = nodes + messages
        out = self.readout(updated).squeeze(-1)             # (B, T, F)
        return out


class TemporalAttention(nn.Module):
    """Additive attention pooling over the LSTM's per-timestep hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):  # lstm_out: (B, T, H)
        scores = self.score(lstm_out).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=-1)                 # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (B, H)
        return context, weights


class WorldModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, n_layers: int = 2,
                 n_stages: int = 6, encoder: str = "lstm", dropout: float = 0.2):
        super().__init__()
        self.n_features = n_features
        self.encoder_type = encoder

        self.feature_gate = FeatureGateAttention(n_features)
        if encoder == "gnn":
            self.gnn = GNNBlock(n_features)

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             num_layers=n_layers, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.temporal_attn = TemporalAttention(hidden_size)

        # Head A: self-supervised transition dynamics P(S_t+1 | S_1..S_t)
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_features)
        )
        # Head B: infiltration forecast (binary, within next K windows)
        self.infil_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        # Head C: MITRE ATT&CK stage classification
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_stages)
        )

    def forward(self, x):  # x: (B, T, F)
        x_gated, feature_weights = self.feature_gate(x)
        if self.encoder_type == "gnn":
            x_gated = x_gated + self.gnn(x_gated)

        lstm_out, (h_n, c_n) = self.lstm(x_gated)
        context, temporal_weights = self.temporal_attn(lstm_out)

        next_state_pred = self.next_state_head(context)
        infil_logit = self.infil_head(context).squeeze(-1)
        stage_logits = self.stage_head(context)

        return {
            "next_state_pred": next_state_pred,
            "infil_logit": infil_logit,
            "stage_logits": stage_logits,
            "temporal_weights": temporal_weights,     # (B, T) -- which past windows mattered
            "feature_weights": feature_weights,        # (B, T, F) -- which features mattered
        }


def world_model_loss(outputs, next_state_y, infil_y, stage_y,
                      w_transition=1.0, w_infil=1.0, w_stage=1.0):
    l_transition = F.mse_loss(outputs["next_state_pred"], next_state_y)
    l_infil = F.binary_cross_entropy_with_logits(outputs["infil_logit"], infil_y.float())
    l_stage = F.cross_entropy(outputs["stage_logits"], stage_y)
    total = w_transition * l_transition + w_infil * l_infil + w_stage * l_stage
    return total, {"transition": l_transition.item(), "infil": l_infil.item(), "stage": l_stage.item()}


if __name__ == "__main__":
    # quick shape sanity check
    B, T, Fdim = 4, 20, 25
    model = WorldModel(n_features=Fdim)
    x = torch.randn(B, T, Fdim)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape))
    next_state_y = torch.randn(B, Fdim)
    infil_y = torch.randint(0, 2, (B,))
    stage_y = torch.randint(0, 6, (B,))
    loss, parts = world_model_loss(out, next_state_y, infil_y, stage_y)
    print("loss:", loss.item(), parts)
