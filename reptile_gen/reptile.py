import torch.nn.functional as F


def reptile_grad(model, inputs, outputs, optimizer, batch=1):
    parameters = list(model.parameters())
    backup = [p.data.clone() for p in parameters]
    res = run_sgd_epoch(model, inputs, outputs, optimizer, batch)
    for b, p in zip(backup, parameters):
        p.grad.copy_(b - p.data)
        p.data.copy_(b)
    return res


def run_sgd_epoch(model, inputs, outputs, optimizer, batch):
    device = next(model.parameters()).device
    losses = []
    for i in range(0, inputs.shape[0], batch):
        x = inputs[i:i+batch]
        y = outputs[i:i+batch]
        out = model(x.to(device))
        target = y.to(device)
        if target.dtype.is_floating_point:
            loss = F.binary_cross_entropy_with_logits(out, target)
        else:
            loss = F.cross_entropy(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
