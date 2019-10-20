import torch
import torch.nn.functional as F


def reptile_grad(model, inputs, outputs, optimizer, inner_iters=1, batch=1):
    parameters = list(model.parameters())
    backup = [p.data.clone() for p in parameters]
    backup_grads = [p.grad.clone() if p.grad is not None else None
                    for p in parameters]
    res = run_sgd_epoch(model, inputs, outputs, optimizer, inner_iters, batch)
    for bg, b, p in zip(backup_grads, backup, parameters):
        if bg is None:
            bg = torch.zeros_like(b)
        p.grad.copy_(bg + b - p.data)
        p.data.copy_(b)
    return res


def run_sgd_epoch(model, inputs, outputs, optimizer, inner_iters, batch):
    device = next(model.parameters()).device
    losses = []
    for i in range(0, inputs.shape[0], batch):
        x = inputs[i:i+batch]
        y = outputs[i:i+batch]
        target = y.to(device)
        for j in range(inner_iters):
            out = model(x.to(device))
            if target.dtype.is_floating_point:
                loss = F.binary_cross_entropy_with_logits(out, target)
            else:
                loss = F.cross_entropy(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if j == 0:
                losses.append(loss.item())
    return losses


def batched_reptile_grad(model, data_points, lr):
    inputs = torch.stack([x for x, _ in data_points])
    outputs = torch.stack([y for _, y in data_points])
    res, final_parameters = batched_run_sgd_epoch(model, inputs, outputs, lr)
    for p, final in zip(model.parameters(), final_parameters):
        g = p.detach() - torch.mean(final, dim=0)
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)
    return res


def batched_run_sgd_epoch(model, inputs, outputs, lr):
    device = next(model.parameters()).device
    parameters = model.batch_parameters(inputs.shape[0])
    losses = []
    for i in range(inputs.shape[1]):
        x = inputs[:, i:i+1].to(device)
        y = outputs[:, i:i+1].to(device)
        out = model.batch_forward(parameters, x)
        if y.dtype.is_floating_point:
            loss = F.binary_cross_entropy_with_logits(out, y)
        else:
            loss = F.cross_entropy(out, y)
        losses.append(loss.item())
        grads = torch.autograd.grad(loss, parameters)
        parameters = tuple((p - lr * g).detach().requires_grad_()
                           for p, g in zip(parameters, grads))
    return losses, parameters
