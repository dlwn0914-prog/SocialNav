"""VQ-VAE over short pedestrian paths (van den Oord et al. 2017, "Neural
Discrete Representation Learning"): a codebook of walking behaviours learned
from data instead of hand-designed features.

What is encoded: the path's DEVIATION from walking straight on at the
current speed, r_k = p(t_k) - (speed * t_k, 0) in the walker's own frame
(x = current heading). Raw positions would split codes mostly by speed;
the deviation keeps what the walker chose to do -- veer left/right, slow
down, speed up. A code's path is recovered as constant velocity + its
decoded deviation, so codes can be drawn and handed to a planner.

Codebook: EMA updates (as in the paper's appendix / VQ-VAE-2) with dead-code
restarts from random encoder outputs of the batch, which keeps all K codes
in use on this small, low-dimensional input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def deviation(future, speed, dt=0.8):
    """future (B, H, 2), speed (B,) -> (B, H, 2) deviation from constant velocity."""
    t = dt * torch.arange(1, future.shape[1] + 1, dtype=future.dtype, device=future.device)
    cv = torch.stack([speed[:, None] * t, torch.zeros_like(speed[:, None] * t)], dim=-1)
    return future - cv


class EMACodebook(nn.Module):
    def __init__(self, num_codes, dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay, self.eps = decay, eps
        self.register_buffer("embed", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_sum", self.embed.clone())

    def assign(self, z):
        return torch.cdist(z, self.embed).argmin(dim=1)

    def forward(self, z):
        idx = self.assign(z)
        q = self.embed[idx]
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.embed.shape[0]).type(z.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_sum.mul_(self.decay).add_(onehot.T @ z, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = (self.cluster_size + self.eps) / (n + self.embed.shape[0] * self.eps) * n
                self.embed.copy_(self.embed_sum / size[:, None])
                dead = self.cluster_size < 1.0
                if dead.any():
                    pick = z[torch.randint(0, len(z), (int(dead.sum()),), device=z.device)]
                    self.embed[dead] = pick
                    self.embed_sum[dead] = pick
                    self.cluster_size[dead] = 1.0
        return q, idx


class TrajVQVAE(nn.Module):
    def __init__(self, horizon=6, num_codes=16, latent=8, hidden=128, beta=0.25):
        super().__init__()
        d = horizon * 2
        self.horizon, self.beta = horizon, beta
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d))
        self.codebook = EMACodebook(num_codes, latent)
        self.register_buffer("scale", torch.ones(1))  # deviation std, set from the training data

    def encode(self, dev):
        return self.enc(dev.flatten(1) / self.scale)

    def decode_codes(self, idx=None):
        """Decoded deviation (K or B, H, 2) for code indices (all codes if None)."""
        e = self.codebook.embed if idx is None else self.codebook.embed[idx]
        return self.dec(e).view(-1, self.horizon, 2) * self.scale

    def forward(self, dev):
        z = self.encode(dev)
        q, idx = self.codebook(z)
        q_st = z + (q - z).detach()  # straight-through
        rec = self.dec(q_st).view(-1, self.horizon, 2) * self.scale
        loss_rec = F.mse_loss(rec / self.scale, dev / self.scale)
        loss_commit = F.mse_loss(z, q.detach())
        return rec, idx, loss_rec + self.beta * loss_commit, loss_rec
