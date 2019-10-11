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
