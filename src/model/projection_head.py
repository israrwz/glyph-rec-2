import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ProjectionHeadConfig:
    """
    Configuration for a two-layer projection MLP.

    Fields:
        input_dim     : Dimension of incoming (base) embeddings.
        hidden_dim    : Hidden layer width.
        output_dim    : Final projected dimension (L2-normalized).
        dropout       : Dropout probability applied after each norm.
        activation    : 'gelu' | 'relu' | 'silu'.
        norm          : 'layer' | 'batch' | 'none'.
        residual      : Whether to add a residual skip from input to output_dim.
                        If input_dim != output_dim and residual=True, a linear
                        projection is applied to match shapes.
    """

    input_dim: int
    hidden_dim: int = 512
    output_dim: int = 128
    dropout: float = 0.1
    activation: str = "gelu"
    norm: str = "layer"
    residual: bool = True


def _make_activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "gelu":
        return nn.GELU()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(kind: str, dim: int) -> nn.Module:
    k = (kind or "none").lower()
    if k == "batch":
        return nn.BatchNorm1d(dim)
    if k == "layer":
        return nn.LayerNorm(dim)
    if k == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {kind}")


class ProjectionHead(nn.Module):
    """
    Two-layer projection MLP for adapting frozen embeddings to a contrastive
    / retrieval-friendly space.

    Architecture:
        x -> Linear(input_dim->hidden_dim) -> Act -> Norm -> Drop
           -> Linear(hidden_dim->output_dim) -> Act -> Norm -> Drop
           -> (optional residual) -> L2 normalize (if norm_last)

    Residual:
        If residual=True and input_dim != output_dim, a linear projection
        matches dimensions; if they match, Identity is used.

    Forward:
        Returns an L2-normalized tensor (B, output_dim) if norm_last=True.
    """

    def __init__(self, cfg: ProjectionHeadConfig, norm_last: bool = True):
        super().__init__()
        self.cfg = cfg
        self.norm_last = norm_last

        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.act1 = _make_activation(cfg.activation)
        self.norm1 = _make_norm(cfg.norm, cfg.hidden_dim)
        self.drop1 = nn.Dropout(cfg.dropout)

        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.act2 = _make_activation(cfg.activation)
        self.norm2 = _make_norm(cfg.norm, cfg.output_dim)
        self.drop2 = nn.Dropout(cfg.dropout)

        if cfg.residual:
            if cfg.input_dim == cfg.output_dim:
                self.residual_proj = nn.Identity()
            else:
                self.residual_proj = nn.Linear(cfg.input_dim, cfg.output_dim)
        else:
            self.residual_proj = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act1(h)
        h = self.norm1(h)
        h = self.drop1(h)

        h2 = self.fc2(h)
        h2 = self.act2(h2)
        h2 = self.norm2(h2)
        h2 = self.drop2(h2)

        if self.residual_proj is not None:
            h2 = h2 + self.residual_proj(x)

        if self.norm_last:
            h2 = torch.nn.functional.normalize(h2, dim=1)
        return h2


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    label_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Supervised contrastive / InfoNCE-style loss.

    For each anchor i:
        Positives: all j != i with labels[j] == labels[i].
        Negatives: all other samples.

    z          : (B, d) expected L2-normalized.
    labels     : (B,) int tensor.
    temperature: Softmax temperature.
    label_weights (optional): Per-sample weighting (B,).

    Returns scalar loss. Samples with no positives are ignored in the average.
    """
    B = z.shape[0]
    if B <= 1:
        return z.new_tensor(0.0)

    # Similarity (cosine since z expected normalized)
    sim = z @ z.t()  # (B,B)
    sim = sim / temperature

    # Exclude self
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(eye, -float("inf"))

    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B,B)
    labels_eq.masked_fill_(eye, False)  # remove self

    # Row-wise stabilization
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_exp = torch.exp(sim - sim_max)

    pos_mask = labels_eq
    pos_sum = (sim_exp * pos_mask).sum(dim=1)
    denom = sim_exp.sum(dim=1) + 1e-12

    # Some rows may have zero positives -> skip in average
    valid_mask = (pos_mask.sum(dim=1) > 0).float()
    loss_vec = -torch.log((pos_sum + 1e-12) / denom)

    if label_weights is not None:
        loss_vec = loss_vec * label_weights

    loss = (loss_vec * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    return loss
