import torch
import torch.nn as nn
import torch.nn.functional as F

from dfm_flow_utils import GaussianFourierProjection


class Dense(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuadCondCNN(nn.Module):
    """
    CNN used for discrete Dirichlet flow matching on DNA sequences.

    Inputs:
      - xt: [B, L, K] or expanded simplex [B, L, 2K]
      - t:  [B] alpha/time
      - cond: [B, cond_dim] (e.g. normalized level)
    Output:
      - logits: [B, L, K]
    """

    def __init__(
        self,
        *,
        alphabet_size: int = 4,
        cond_dim: int = 1,
        hidden_dim: int = 256,
        num_cnn_stacks: int = 2,
        dropout: float = 0.1,
        expanded_simplex: bool = True,
        time_embed_scale: float = 1.0,
    ):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_cnn_stacks = int(num_cnn_stacks)
        self.dropout_p = float(dropout)
        self.expanded_simplex = bool(expanded_simplex)

        inp_size = self.alphabet_size * (2 if self.expanded_simplex else 1)

        self.in_proj = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)

        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(embedding_dim=self.hidden_dim, scale=time_embed_scale),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_embedder = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        base_convs = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256),
        ]
        self.num_layers = len(base_convs) * self.num_cnn_stacks
        self.convs = nn.ModuleList([layer for _ in range(self.num_cnn_stacks) for layer in base_convs])
        self.time_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.cond_layers = nn.ModuleList([Dense(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(self.dropout_p)

        self.out = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.alphabet_size, kernel_size=1),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # xt: [B, L, C] -> [B, C, L]
        time_emb = F.relu(self.time_embedder(t))
        cond_emb = self.cond_embedder(cond)

        h = xt.permute(0, 2, 1)
        h = F.relu(self.in_proj(h))

        for i in range(self.num_layers):
            z = self.dropout(h.clone())
            z = z + self.time_layers[i](time_emb)[:, :, None]
            z = z + self.cond_layers[i](cond_emb)[:, :, None]
            z = self.norms[i](z.permute(0, 2, 1)).permute(0, 2, 1)
            z = F.relu(self.convs[i](z))
            h = h + z if z.shape == h.shape else z

        y = self.out(h)  # [B, K, L]
        return y.permute(0, 2, 1)  # [B, L, K]

