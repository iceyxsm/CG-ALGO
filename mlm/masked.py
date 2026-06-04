"""Masked-candle pretraining: the language-model move.

Pretrain the encoder self-supervised by masking candle tokens and reconstructing
their feature vectors, so it learns candle structure from far more (unlabeled)
data than the sparse WIN/LOSS labels provide. Then fine-tune for direction.

To keep the cross-asset transfer claim clean, pretraining uses dense overlapping
windows from the training asset's train-region only: the evaluation test split
and any held-out target asset are never seen during pretraining. If this beats
the from-scratch transformer's ~0.59 dir_auc, representation learning unlocked
signal the labels alone could not; if it matches, the signal is local and fully
captured, and stage one is closed.
"""
import numpy as np
import torch
import torch.nn as nn

from .features import compute_features, FeatureConfig, FEATURE_COLUMNS
from .labeling import WIN
from .transformer import _to_tensor, _C, _DEVICE

_D = 64


class _Encoder(nn.Module):
    def __init__(self, channels, d_model=_D, heads=4, layers=2, max_len=512):
        super().__init__()
        self.embed = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 2,
                                         batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)

    def forward(self, x):
        return self.enc(self.embed(x) + self.pos[:, :x.size(1)])


class _Recon(nn.Module):
    def __init__(self, enc, channels, d_model=_D):
        super().__init__()
        self.enc, self.out = enc, nn.Linear(d_model, channels)

    def forward(self, x):
        return self.out(self.enc(x))


class _Clf(nn.Module):
    def __init__(self, enc, d_model=_D):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        return self.head(self.enc(x).mean(1)).squeeze(-1)


def dense_windows(df, window=64, end_frac=0.7, stride=2, max_windows=200000,
                  feat_cfg=FeatureConfig(), seed=0):
    """Dense overlapping feature windows from the first `end_frac` of the series.

    Overlapping (stride 2) so pretraining sees far more windows than the
    non-overlapping labeled sampler; capped at `end_frac` so it never reaches the
    evaluation test split. Returns flattened rows matching the model interface.
    """
    feats = compute_features(df, feat_cfg)
    fvals = feats[FEATURE_COLUMNS].to_numpy()
    valid = feats["valid"].to_numpy()
    start = int(np.argmax(valid)) + window
    end = int(len(df) * end_frac)
    W = [fvals[e - window:e] for e in range(start, end, stride)
         if np.isfinite(fvals[e - window:e]).all()]
    X = np.stack(W).reshape(len(W), -1)
    if max_windows and len(X) > max_windows:
        sel = np.random.default_rng(seed).choice(len(X), max_windows, False)
        X = X[sel]
    return X


def pretrain_encoder(pretrain_X, epochs=8, batch=256, mask_frac=0.15, lr=5e-4,
                     seed=0, verbose=True):
    """Masked-candle reconstruction; returns the trained encoder."""
    torch.manual_seed(seed)
    enc = _Encoder(_C).to(_DEVICE)
    model = _Recon(enc, _C).to(_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_all = _to_tensor(pretrain_X).to(_DEVICE)
    if verbose:
        print(f"  pretrain device={_DEVICE} windows={len(x_all):,}", flush=True)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(x_all), device=_DEVICE)
        tot, nb = 0.0, 0
        for i in range(0, len(x_all), batch):
            xb = x_all[perm[i:i + batch]]
            mask = torch.rand(xb.shape[:2], device=_DEVICE) < mask_frac
            xin = xb.clone()
            xin[mask] = 0.0
            pred = model(xin)
            loss = ((pred - xb)[mask] ** 2).mean() if mask.any() \
                else (pred - xb).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if verbose:
            print(f"  pretrain epoch {ep + 1}/{epochs} mse {tot / nb:.4f}",
                  flush=True)
    return enc


def train_masked(train, val, pretrain_X=None, epochs=20, batch=256, lr=5e-4,
                 seed=0, verbose=True):
    """Pretrain the encoder (masked reconstruction) then fine-tune on WIN-vs-not.

    pretrain_X defaults to the fine-tune training windows; pass dense_windows of
    the training asset for the full more-data pretraining test.
    """
    torch.manual_seed(seed)
    enc = pretrain_encoder(pretrain_X if pretrain_X is not None else train[0],
                           seed=seed, verbose=verbose)
    model = _Clf(enc).to(_DEVICE)
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    xtr = _to_tensor(Xtr).to(_DEVICE)
    ytr_b = torch.tensor((ytr == WIN).astype(np.float32)).to(_DEVICE)
    xva = _to_tensor(Xva).to(_DEVICE)
    yva_b = torch.tensor((yva == WIN).astype(np.float32)).to(_DEVICE)
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
            print(f"  finetune epoch {ep + 1}/{epochs} val_logloss {vloss:.4f}",
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


def predict_masked(model, X, batch=1024):
    """Return P(WIN) per row, matching predict_win_prob's interface."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(
                model(_to_tensor(X[i:i + batch]).to(_DEVICE))).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
