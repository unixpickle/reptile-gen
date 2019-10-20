"""
Models for generating images.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embed = nn.Embedding(28, 128)
        self.y_embed = nn.Embedding(28, 128)
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x_vec = self.x_embed(x[:, 0])
        y_vec = self.y_embed(x[:, 1])
        return self.layers(torch.cat([x_vec, y_vec], dim=-1))


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


class BatchModule(ABC, nn.Module):
    """
    A model which can be applied with a batch of
    parameters to a batch of input batches.
    """

    def __init__(self):
        nn.Module.__init__(self)

    def batch_parameters(self, batch):
        """
        Repeat the parameters from self.parameters() the
        given number of times.

        Args:
            batch: the number of times to repeat each
              parameter in the batch dimension.

        Returns:
            A tuple of batched parameters.
        """
        return tuple(x[None].repeat(batch, *([1]*len(x.shape))) for x in self.parameters())

    @abstractmethod
    def batch_forward(self, parameters, xs):
        """
        Apply the model with a batch of parameters and a
        batch of input batches.

        Args:
            parameters: a tuple of Tensors where each
              Tensor is a batch for a given parameter.
            xs: a batch of input batches.

        Returns:
            A batch of output batches.
        """
        pass


class BatchSequential(BatchModule):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(self.layers):
            self.add_module('module_%d' % i, layer)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def batch_forward(self, parameters, xs):
        i = 0
        for l in self.layers:
            num_params = len(list(l.parameters()))
            ps = parameters[i:i+num_params]
            i += num_params
            xs = l.batch_forward(ps, xs)
        return xs


class BatchFn(BatchModule):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

    def batch_forward(self, parameters, xs):
        return self.fn(xs)


class BatchEmbedding(BatchModule):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.table = nn.Parameter(torch.randn(num_inputs, num_outputs))

    def forward(self, x):
        return F.embedding(x, self.table)

    def batch_forward(self, parameters, xs):
        table = parameters[0]
        outputs = []
        for i in range(xs.shape[0]):
            outputs.append(F.embedding(xs[i], table[i]))
        return torch.stack(outputs, dim=0)


class BatchMultiEmbedding(BatchModule):
    def __init__(self, *embeddings):
        super().__init__()
        self.embeddings = embeddings
        for i, layer in enumerate(embeddings):
            self.add_module('module_%d' % i, layer)

    def forward(self, x):
        return torch.cat([l(x[..., i]) for i, l in enumerate(self.embeddings)], dim=-1)

    def batch_forward(self, parameters, xs):
        outputs = []
        for i, (param, layer) in enumerate(zip(parameters, self.embeddings)):
            outputs.append(layer.batch_forward((param,), xs[..., i]))
        return torch.cat(outputs, dim=-1)


class BatchLinear(BatchModule):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear(x)

    def batch_forward(self, parameters, xs):
        weight, bias = parameters
        output = torch.bmm(xs, weight.permute(0, 2, 1))
        output = output + bias[:, None]
        return output


def batch_mnist_model():
    return BatchSequential(
        BatchMultiEmbedding(
            BatchEmbedding(28, 128),
            BatchEmbedding(28, 128),
        ),
        BatchLinear(256, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 1),
    )
