"""
train.py
---------
Trains Zynex's World Model (LSTM/GNN + attention, multi-task heads) on the
sequences produced by feature_extraction.py.

Usage:
    python src/train.py --data data/features.npz --epochs 15 --out models/world_model.pt
    python src/train.py --data data/features.npz --encoder gnn --epochs 15 --out models/world_model.pt
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from world_model import WorldModel, world_model_loss


def load_data(npz_path, test_frac=0.2):
    d = np.load(npz_path, allow_pickle=True)
    seq_X = d["seq_X"]
    next_state_y = d["next_state_y"]
    infil_y = d["infil_y"]
    stage_y = d["stage_y"]
    mean, std = d["mean"], d["std"]
    feature_names = d["feature_names"]

    n = len(seq_X)
    split = int(n * (1 - test_frac))

    train_ds = TensorDataset(
        torch.tensor(seq_X[:split]), torch.tensor(next_state_y[:split]),
        torch.tensor(infil_y[:split]), torch.tensor(stage_y[:split]),
    )
    test_ds = TensorDataset(
        torch.tensor(seq_X[split:]), torch.tensor(next_state_y[split:]),
        torch.tensor(infil_y[split:]), torch.tensor(stage_y[split:]),
    )
    return train_ds, test_ds, mean, std, feature_names, seq_X.shape[-1]


def evaluate_loader(model, loader, device):
    model.eval()
    correct_infil, total, infil_probs, infil_true = 0, 0, [], []
    with torch.no_grad():
        for x, next_y, infil_y, stage_y in loader:
            x, next_y, infil_y, stage_y = x.to(device), next_y.to(device), infil_y.to(device), stage_y.to(device)
            out = model(x)
            probs = torch.sigmoid(out["infil_logit"])
            preds = (probs > 0.5).long()
            correct_infil += (preds == infil_y).sum().item()
            total += len(infil_y)
            infil_probs.append(probs.cpu().numpy())
            infil_true.append(infil_y.cpu().numpy())
    acc = correct_infil / max(total, 1)
    return acc, np.concatenate(infil_probs), np.concatenate(infil_true)


def main():
    ap = argparse.ArgumentParser(description="Train Zynex World Model")
    ap.add_argument("--data", type=str, default="data/features.npz")
    ap.add_argument("--out", type=str, default="models/world_model.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--encoder", type=str, default="lstm", choices=["lstm", "gnn"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, mean, std, feature_names, n_features = load_data(args.data, args.test_frac)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = WorldModel(n_features=n_features, hidden_size=args.hidden_size,
                        n_layers=args.n_layers, encoder=args.encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, next_y, infil_y, stage_y in train_loader:
            x, next_y, infil_y, stage_y = x.to(device), next_y.to(device), infil_y.to(device), stage_y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss, parts = world_model_loss(out, next_y, infil_y, stage_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x)
        epoch_loss /= len(train_ds)

        test_acc, _, _ = evaluate_loader(model, test_loader, device)
        print(f"[train] epoch {epoch:02d}/{args.epochs}  loss={epoch_loss:.4f}  test_infil_acc={test_acc:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "n_features": n_features,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "encoder": args.encoder,
        "mean": mean, "std": std,
        "feature_names": feature_names,
    }, args.out)
    print(f"[train] saved model -> {args.out}")


if __name__ == "__main__":
    main()
