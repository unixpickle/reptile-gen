import math

import torch

from .reptile import inferred_loss


def maml_grad(model, data_points, lr, checkpoint=True):
    """
    Compute the model's meta-gradient using MAML.

    See reptile_grad for more details on usage.

    Args:
        model: a BatchModel to compute a gradient for.
        data_points: a meta-batch of (inputs, outputs).
        lr: the inner-loop learning rate.
        checkpoint: if True, conserve memory at the cost
          of a little extra compute.

    Returns:
        A list of losses, one per inner-loop step.
    """
    batch_size = data_points[0][0].shape[0]

    # Adjust for fact that we average over the
    # whole meta-batch.
    lr *= batch_size

    device = next(model.parameters()).device
    inputs = torch.stack([x for x, _ in data_points]).to(device)
    outputs = torch.stack([y for _, y in data_points]).to(device)

    init_params = model.batch_parameters(batch_size)
    if not checkpoint:
        gradient, losses = _maml_grad(model, init_params, inputs, outputs, lr,
                                      [torch.zeros_like(p) for p in init_params])
    else:
        gradient, losses = _checkpointed_maml_grad(model, init_params, inputs, outputs, lr)
    for p, g in zip(model.parameters(), gradient):
        avg_grad = torch.sum(g, dim=0)
        if p.grad is None:
            p.grad = avg_grad
        else:
            p.grad.add_(avg_grad)
    return losses


def _checkpointed_maml_grad(model, parameters, inputs, outputs, lr):
    num_steps = inputs.shape[1]
    interval = int(math.sqrt(num_steps))
    checkpoints = []
    losses = []
    params = parameters
    for i in range(num_steps):
        x = inputs[:, i:i+1]
        y = outputs[:, i:i+1]
        if i % interval == 0:
            checkpoints.append(tuple(p.clone().detach().requires_grad for p in params))
        out = model.batch_forward(params, x)
        loss = inferred_loss(out, y)
        losses.append(loss.item())
        grads = torch.autograd.grad(loss, params)
        params = tuple((p - lr * g).detach() for p, g in zip(params, grads))
    gradient = [torch.zeros_like(p) for p in params]
    for i in list(range(0, num_steps, interval))[::-1]:
        checkpoint = checkpoints[i // interval]
        gradient, _ = _maml_grad(model,
                                 checkpoint,
                                 inputs[:, i:i+interval],
                                 outputs[:, i:i+interval],
                                 lr,
                                 gradient)
    return gradient, losses


def _maml_grad(model, init_params, inputs, outputs, lr, grad_outputs):
    losses = []
    total_loss = 0
    parameters = init_params
    for i in range(inputs.shape[1]):
        x = inputs[:, i:i+1]
        y = outputs[:, i:i+1]
        out = model.batch_forward(parameters, x)
        loss = inferred_loss(out, y)
        losses.append(loss.item())
        grads = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
        parameters = tuple(p - lr * g for p, g in zip(parameters, grads))
        total_loss = total_loss + loss
    gradient = torch.autograd.grad(total_loss, init_params, grad_outputs=grad_outputs)
    return gradient, losses
