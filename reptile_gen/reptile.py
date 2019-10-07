import functools

import torch
import torch.optim as optim
import torch.nn.functional as F


def reptile_step(model, inputs, outputs, lr=1e-3, epsilon=0.1):
    train_fn = functools.partial(run_sgd_epoch, model, inputs, outputs, lr)
    return interpolate_parameters(model.parameters(), epsilon, train_fn)


def interpolate_parameters(parameters, epsilon, fn):
    parameters = list(parameters)
    backup = [p.data.clone() for p in parameters]
    res = fn()
    for b, p in zip(backup, parameters):
        p.data.copy_(b + epsilon * (p - b))
    return res


def run_sgd_epoch(model, inputs, outputs, lr):
    device = next(model.parameters()).device
    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []
    for x, y in zip(inputs, outputs):
        out = model(torch.from_numpy(x[None]).to(device).float())
        target = torch.from_numpy(y[None]).to(device).float()
        loss = F.binary_cross_entropy_with_logits(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
