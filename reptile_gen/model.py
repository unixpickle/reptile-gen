"""
Models for generating images.
"""

import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embed = nn.Embedding(28, 32)
        self.y_embed = nn.Embedding(28, 32)
        self.layers = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x_vec = self.x_embed(x[:, 0])
        y_vec = self.y_embed(x[:, 1])
        return self.layers(torch.cat([x_vec, y_vec], dim=-1))
