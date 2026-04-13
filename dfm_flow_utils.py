from __future__ import annotations

import math

import numpy as np
import scipy
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


def simplex_proj(seq: torch.Tensor) -> torch.Tensor:
    y = seq.reshape(-1, seq.shape[-1])
    n, k = y.shape
    x, _ = torch.sort(y, dim=-1, descending=True)
    x_cumsum = torch.cumsum(x, dim=-1) - 1
    div_seq = torch.arange(1, k + 1, dtype=y.dtype, device=y.device)
    xtmp = x_cumsum / div_seq.unsqueeze(0)
    greater_than_xtmp = (x > xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(n, dtype=torch.long, device=y.device).unsqueeze(1)
    selected_xtmp = xtmp[row_indices, greater_than_xtmp - 1]
    xproj = torch.max(y - selected_xtmp, torch.zeros_like(y))
    return xproj.view(seq.shape)


def sample_cond_prob_path(
    seq: torch.Tensor,
    alphabet_size: int,
    *,
    alpha_scale: float = 15.0,
    alpha_max: float = 12.0,
    fix_alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, l = seq.shape
    seq_one_hot = F.one_hot(seq, num_classes=alphabet_size).float()
    if fix_alpha is not None:
        alphas = torch.ones(b, device=seq.device, dtype=torch.float32) * float(fix_alpha)
    else:
        draws = scipy.stats.expon().rvs(size=b).astype(np.float32)
        alphas = torch.from_numpy(1.0 + draws * float(alpha_scale)).to(seq.device)
        alphas = torch.clamp(alphas, max=float(alpha_max))
    alphas_ = torch.ones(b, l, alphabet_size, device=seq.device, dtype=torch.float32)
    alphas_ = alphas_ + seq_one_hot * (alphas[:, None, None] - 1)
    xt = torch.distributions.Dirichlet(alphas_).sample()
    return xt, alphas


def expand_simplex(
    xt: torch.Tensor,
    alphas: torch.Tensor,
    prior_pseudocount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_weights = (prior_pseudocount / (alphas + prior_pseudocount - 1))[:, None, None]
    expanded = torch.cat([xt * (1 - prior_weights), xt * prior_weights], dim=-1)
    return expanded, prior_weights


class DirichletConditionalFlow:

    def __init__(
        self,
        k: int,
        alpha_min: float = 1.0,
        alpha_max: float = 12.0,
        alpha_spacing: float = 0.001,
    ):
        self.k = k
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing, dtype=np.float64)
        self.bs = np.linspace(0, 1, 1000, dtype=np.float64)
        self.beta_cdfs = []
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, k - 1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing

    def c_factor(self, bs: np.ndarray, alpha: float) -> np.ndarray:
        out1 = scipy.special.beta(alpha, self.k - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.k - 1)), 0)
        out = np.where((bs ** (alpha - 1)) > 0, out2 / (bs ** (alpha - 1)), 0)
        i_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, i_func)
        return interp * out


class GaussianFourierProjection(nn.Module):

    def __init__(self, embedding_dim: int = 256, scale: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim // 2) * scale, requires_grad=False)
        self.embedding_dim = embedding_dim

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        shape = signal.shape
        signal = signal.view(-1)
        signal_proj = signal[:, None] * self.W[None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(signal_proj), torch.cos(signal_proj)], dim=-1)
        return emb.view(*shape, self.embedding_dim)
