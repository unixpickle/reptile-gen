import math

import torch
import torch.nn.functional as F


def maml_grad(model, inputs, outputs, lr, batch=1, checkpoint=False):
    """
    Update the model gradient using MAML.
    """
    params = list(model.parameters())
    device = params[0].device
    batches = list(_split_batches(inputs.to(device), outputs.to(device), batch))
    if outputs.dtype.is_floating_point:
        loss_fn = F.binary_cross_entropy_with_logits
    else:
        loss_fn = F.cross_entropy
    if not checkpoint:
        gradient, losses = _maml_grad(model, batches, lr, loss_fn,
                                      [torch.zeros_like(p) for p in params])
    else:
        gradient, losses = _checkpointed_maml_grad(model, batches, lr, loss_fn)
    for p, g in zip(params, gradient):
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)
    return losses


def _split_batches(inputs, outputs, batch):
    for i in range(0, inputs.shape[0], batch):
        yield (inputs[i:i+batch], outputs[i:i+batch])


def _checkpointed_maml_grad(model, batches, lr, loss_fn):
    params = list(model.parameters())
    interval = int(math.sqrt(len(batches)))
    checkpoints = []
    scalar_losses = []
    for i, (x, y) in enumerate(batches):
        if i % interval == 0:
            checkpoints.append([p.clone().detach() for p in params])
        out = model(x)
        loss = loss_fn(out, y)
        scalar_losses.append(loss.item())
        grads = torch.autograd.grad(loss, params)
        for p, g in zip(params, grads):
            p.data.add_(-lr * g)
    gradient = [torch.zeros_like(p) for p in params]
    for i in list(range(0, len(batches), interval))[::-1]:
        checkpoint = checkpoints[i // interval]
        for p, v in zip(params, checkpoint):
            p.data.copy_(v)
        gradient, _ = _maml_grad(model, batches[i:i+interval], lr, loss_fn, gradient)
    return gradient, scalar_losses


def _maml_grad(model, batches, lr, loss_fn, grad_outputs):
    params = list(model.parameters())
    initial_values = []
    final_values = []
    loss_grads = []
    scalar_losses = []
    for x, y in batches:
        out = model(x)
        loss = loss_fn(out, y)
        scalar_losses.append(loss.item())
        initial_values.append([p.clone().detach() for p in params])
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        loss_grads.append([g.detach() for g in grads])
        updated = []
        for grad, param in zip(grads, params):
            x = param - lr * grad
            updated.append(x)
            param.data.copy_(x)
        final_values.append(updated)
    gradient = grad_outputs
    for loss_grad, initial, final in list(zip(loss_grads, initial_values, final_values))[::-1]:
        for p, x in zip(params, initial):
            p.data.copy_(x)
        future_grad = torch.autograd.grad(final, params, grad_outputs=gradient, retain_graph=True)
        gradient = [v1 + v2 for v1, v2 in zip(loss_grad, future_grad)]
    return gradient, scalar_losses
