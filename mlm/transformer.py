"""Transformer rung: tests whether long-range structure across the window carries
signal beyond what local motifs (CNN) and flattened positions (tree) capture.

Each candle is a token; a learned positional encoding preserves order, and
self-attention lets any candle attend to any other across the full window. This
is the model built for the "grammar" hypothesis: if dir_auc rises meaningfully
above the CNN with the same stability, sequential long-range structure matters.
Same (X, y, idx) interface and predict signature as baseline.py / cnn.py, so it
drops into transfer_matrix via fit=/predict=.
"""
import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_COLUMNS
from .labeling import WIN

_C = len(FEATURE_COLUMNS)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _Transformer(nn.Module):
    def __init__(self, channels, d_model=64, heads=4, layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 2,
                                         batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                 # x: (N, time, C)
        h = self.embed(x) + self.pos[:, :x.size(1)]
        h = self.enc(h).mean(dim=1)       # pool over time
        return self.head(h).squeeze(-1)


def _to_tensor(X):
    """Flattened rows (N, time*C) -> (N, time, C) tokens for the transformer."""
    n = X.shape[0]
    t = X.shape[1] // _C
    x = X.reshape(n, t, _C)
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)


def train_transformer(train, val, epochs=20, batch=256, lr=5e-4, seed=0,
                      verbose=True):
    """Train on WIN-vs-not, early-stopping on validation logloss."""
    torch.manual_seed(seed)
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    xtr = _to_tensor(Xtr).to(_DEVICE)
    ytr_b = torch.tensor((ytr == WIN).astype(np.float32)).to(_DEVICE)
    xva = _to_tensor(Xva).to(_DEVICE)
    yva_b = torch.tensor((yva == WIN).astype(np.float32)).to(_DEVICE)

    model = _Transformer(_C).to(_DEVICE)
    if verbose:
        print(f"  device={_DEVICE} train={len(xtr):,} val={len(xva):,}",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait, patience = float("inf"), None, 0, 4
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), device=_DEVICE)
        for i in range(0, len(xtr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss_fn(model(xtr[b]), ytr_b[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(xva), yva_b).item()
        if verbose:
            print(f"  epoch {ep + 1}/{epochs}  val_logloss {vloss:.4f}",
                  flush=True)
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, {k: v.clone() for k, v in
                                             model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict_transformer(model, X, batch=1024):
    """Return P(WIN) per row, matching predict_win_prob's interface."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = _to_tensor(X[i:i + batch]).to(_DEVICE)
            out.append(torch.sigmoid(model(x)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
