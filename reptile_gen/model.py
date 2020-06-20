"""
Various models for using Reptile as a sequence model.
"""

from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_text_model(max_len=128):
    return BatchSequential(
        BatchMultiEmbedding(
            BatchEmbedding(max_len, 512),
        ),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 512),
        BatchFn(F.relu),
        BatchLinear(512, 256),
    )


def make_mnist_model():
    return BatchSequential(
        BatchMultiEmbedding(
            BatchEmbedding(28, 128),
            BatchEmbedding(28, 128),
        ),
        BatchLinear(256, 256),
        BatchLayerNorm(256),
        BatchLSTM(256),
        BatchLayerNorm(256),
        BatchLSTM(256),
        BatchLayerNorm(256),
        BatchLinear(256, 1),
    )


def make_mnist_model_siren():
    return BatchSequential(
        BatchDiscreteToContinuous(28),
        BatchSIREN(2, 256),
        BatchSIREN(256, 256),
        BatchSIREN(256, 256),
        BatchSIREN(256, 1),
    )


def gated_act(x):
    """
    A gated activation function designed to allow a model
    to "remember" certain pieces of information by
    preventing that information from being used in the
    forward pass.
    """
    d = x.shape[-1] // 2
    return x[..., :d] * torch.sigmoid(x[..., d:])


class MNISTBaseline(nn.Module):
    """
    A baseline RNN model for MNIST sequence generation.
    """

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


class BatchModule(ABC, nn.Module):
    """
    A model which can be applied with a batch of
    parameters to a batch of input batches.

    These modules can take advantage of batch matrix
    multiplies and other batched operations.
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


class BatchLayerNorm(BatchModule):
    def __init__(self, inner_dim):
        super().__init__()
        self.inner_dim = inner_dim
        self.norm = nn.LayerNorm((inner_dim,))

    def forward(self, x):
        return self.norm(x)

    def batch_forward(self, parameters, xs):
        res = F.layer_norm(xs, (self.inner_dim,))
        res *= parameters[0][:, None]
        res += parameters[1][:, None]
        return res


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


class BatchDiscreteToContinuous(BatchModule):
    def __init__(self, num_values):
        super().__init__()
        self.num_values = num_values

    def forward(self, x):
        return x.float() / (self.num_values / 2) - 1

    def batch_forward(self, parameters, xs):
        return xs.float() / (self.num_values / 2) - 1


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


class BatchSIREN(BatchModule):
    """
    A layer from a SIREN: https://arxiv.org/abs/2006.09661.
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

        min_val = -math.sqrt(6 / num_inputs)
        max_val = -min_val
        init_weight = torch.rand_like(self.linear.weight) * (max_val - min_val) + min_val
        self.linear.weight.detach().copy_(init_weight)
        self.linear.bias.detach().zero_()

    def forward(self, x):
        return torch.sin(self.linear(x))

    def batch_forward(self, parameters, xs):
        weight, bias = parameters
        output = torch.bmm(xs, weight.permute(0, 2, 1))
        output = output + bias[:, None]
        return torch.sin(output)


class BatchResidual(BatchSequential):
    def forward(self, x):
        return super().forward(x) + x

    def batch_forward(self, parameters, xs):
        return super().batch_forward(parameters, xs) + xs


class BatchGatedResidual(BatchSequential):
    def __init__(self, in_size, *layers):
        super().__init__(*layers)
        self.weights = BatchLinear(in_size, in_size)

    def forward(self, x):
        gates = self.weights(x)
        outs = super().forward(x)
        return gated_act(torch.cat([outs, gates], dim=-1))

    def batch_forward(self, parameters, xs):
        gates = self.weights.batch_forward(parameters[-2:], xs)
        outs = super().batch_forward(parameters[:-2], xs)
        return gated_act(torch.cat([outs, gates], dim=-1))


class BatchLSTM(BatchModule):
    def __init__(self, dim):
        super().__init__()
        self.hidden_vec = nn.Parameter(torch.randn(dim))
        self.forget_gate = BatchLinear(dim * 2, dim)
        self.input_gate = BatchLinear(dim * 2, dim)
        self.output_gate = BatchLinear(dim * 2, dim)
        self.updater = BatchLinear(dim * 2, dim)

    def forward(self, x):
        hiddens = self.hidden_vec[None].repeat(x.shape[0], 1)
        joined_in = torch.cat([hiddens, x], dim=-1)
        forget = torch.sigmoid(self.forget_gate(joined_in))
        inputs = torch.sigmoid(self.input_gate(joined_in))
        outputs = torch.sigmoid(self.output_gate(joined_in))
        updates = self.updater(joined_in)
        new_hidden = inputs * torch.tanh(updates) + forget * hiddens
        return outputs * new_hidden

    def batch_forward(self, parameters, xs):
        hiddens = parameters[0][:, None].repeat(1, xs.shape[1], 1)
        joined_in = torch.cat([hiddens, xs], dim=-1)
        forget = torch.sigmoid(self.forget_gate.batch_forward(parameters[1:3], joined_in))
        inputs = torch.sigmoid(self.input_gate.batch_forward(parameters[3:5], joined_in))
        outputs = torch.sigmoid(self.output_gate.batch_forward(parameters[5:7], joined_in))
        updates = self.updater.batch_forward(parameters[7:9], joined_in)
        new_hidden = inputs * torch.tanh(updates) + forget * hiddens
        return outputs * new_hidden
