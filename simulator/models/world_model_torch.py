from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class WorldModelConfig:
    hidden_sizes: list[int]
    device: str

def onehot_action(a: torch.Tensor, n_actions: int = 4) -> torch.Tensor:
    # a: [B] int64
    return torch.nn.functional.one_hot(a.long(), num_classes=n_actions).float()

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class WorldModel(nn.Module):
    """Predicts:
    - next state (E',F',P') in [0,1] via sigmoid
    - purchase probability via sigmoid
    - churn probability via sigmoid
    """
    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = 3 + 4
        self.backbone = MLP(in_dim, cfg.hidden_sizes, out_dim=64)
        self.head_state = nn.Linear(64, 3)
        self.head_buy = nn.Linear(64, 1)
        self.head_churn = nn.Linear(64, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, onehot_action(a)], dim=-1)
        h = self.backbone(x)
        s_next = torch.sigmoid(self.head_state(h))
        p_buy = torch.sigmoid(self.head_buy(h)).squeeze(-1)
        p_churn = torch.sigmoid(self.head_churn(h)).squeeze(-1)
        return s_next, p_buy, p_churn
