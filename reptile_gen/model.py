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


class MNISTBaseline(nn.Module):
    def __init__(self, layers=3, stop_grad=False):
        super().__init__()
        self.stop_grad = stop_grad
        self.x_embed = nn.Embedding(28, 128)
        self.y_embed = nn.Embedding(28, 128)
        self.prev_embed = nn.Embedding(2, 128)
        self.rnn = nn.LSTM(128*3, 256, num_layers=layers)
        self.out_layer = nn.Linear(256, 1)
        for i in range(2):
            p = nn.Parameter(torch.zeros([layers, 256], dtype=torch.float))
            self.register_parameter('hidden_%d' % i, p)

    def forward(self, inputs, hidden=None):
        x_vec = self.x_embed(inputs[:, :, 0])
        y_vec = self.y_embed(inputs[:, :, 1])
        prev_vec = self.prev_embed(inputs[:, :, 2])
        x = torch.cat([x_vec, y_vec, prev_vec], dim=-1)

        batch = x.shape[1]
        if hidden is None:
            init_hidden = (self.hidden_0, self.hidden_1)
            hidden = tuple(h[:, None].repeat(1, batch, 1) for h in init_hidden)

        outputs = []
        for t in range(inputs.shape[0]):
            outs, hidden = self.rnn(x[t:t+1], hidden)
            outs = self.out_layer(outs.view(batch, -1)).view(1, batch, 1)
            outputs.append(outs)
            if self.stop_grad:
                hidden = tuple(h.detach() for h in hidden)

        return torch.cat(outputs, dim=0), hidden


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
