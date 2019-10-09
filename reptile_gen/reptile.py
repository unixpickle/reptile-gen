import functools

import torch
import torch.nn.functional as F


def reptile_step(model, inputs, outputs, optimizer, epsilon=0.1):
    train_fn = functools.partial(run_sgd_epoch, model, inputs, outputs, optimizer)
    return interpolate_parameters(model.parameters(), epsilon, train_fn)


def interpolate_parameters(parameters, epsilon, fn):
    parameters = list(parameters)
    backup = [p.data.clone() for p in parameters]
    res = fn()
    for b, p in zip(backup, parameters):
        p.data.copy_(b + epsilon * (p - b))
    return res


def run_sgd_epoch(model, inputs, outputs, optimizer, batch=1):
    device = next(model.parameters()).device
    losses = []
    for i in range(0, len(inputs), batch):
        x = inputs[i:i+batch]
        y = outputs[i:i+batch]
        out = model(torch.from_numpy(x).to(device).long())
        target = torch.from_numpy(y).to(device).float()
        loss = F.binary_cross_entropy_with_logits(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
