"""
Models for generating images.
"""

import torch
import torch.nn as nn


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embed = nn.Embedding(28, 128)
        self.y_embed = nn.Embedding(28, 128)
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x_vec = self.x_embed(x[:, 0])
        y_vec = self.y_embed(x[:, 1])
        return self.layers(torch.cat([x_vec, y_vec], dim=-1))


class TextModel(nn.Module):
    def __init__(self, max_len=128):
        super().__init__()
        self.t_embed = nn.Embedding(max_len, 512)
        self.layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        t_vec = self.t_embed(x[:, 0])
        return self.layers(t_vec)
