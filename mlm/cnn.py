"""1D-CNN rung: tests whether translation-invariant local motifs carry signal.

The LightGBM baseline flattens the window, so it must relearn a pattern at every
position. A 1D convolution slides the same filters across time, so a motif
(compression before expansion, a rejection wick, three strong bodies) is detected
wherever it occurs. If this beats the tree, locality mattered; if it too lands at
zero on a healthy trade count, that is stronger evidence the representation lacks
signal. Same (X, y, idx) interface and predict signature as baseline.py, so it
drops into transfer_matrix via fit=/predict=.
"""
import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_COLUMNS
from .labeling import WIN

_C = len(FEATURE_COLUMNS)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _CNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _to_tensor(X):
    """Flattened rows (N, time*C) -> (N, C, time) for Conv1d."""
    n = X.shape[0]
    t = X.shape[1] // _C
    x = X.reshape(n, t, _C).transpose(0, 2, 1)   # (N, C, time)
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32)


def train_cnn(train, val, epochs=15, batch=256, lr=1e-3, seed=0):
    """Train the CNN on WIN-vs-not, early-stopping on validation logloss."""
    torch.manual_seed(seed)
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    xtr, ytr_b = _to_tensor(Xtr).to(_DEVICE), torch.tensor(
        (ytr == WIN).astype(np.float32)).to(_DEVICE)
    xva, yva_b = _to_tensor(Xva).to(_DEVICE), torch.tensor(
        (yva == WIN).astype(np.float32)).to(_DEVICE)

    model = _CNN(_C).to(_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, patience = float("inf"), None, 4
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


def predict_cnn(model, X):
    """Return P(WIN) per row, matching predict_win_prob's interface."""
    model.eval()
    with torch.no_grad():
        x = _to_tensor(X).to(_DEVICE)
        return torch.sigmoid(model(x)).cpu().numpy()
