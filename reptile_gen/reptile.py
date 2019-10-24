import torch
import torch.nn.functional as F


def reptile_grad(model, data_points, lr, loss_fn=None):
    """
    Compute the model's meta-gradient using Reptile.

    Args:
        model: a BatchModel to compute a gradient for.
        data_points: a meta-batch of (inputs, outputs)
          pairs, where each inputs and outputs is a
          sequence to feed to the model step by step.
          For example, if inputs is of shape [512, 256],
          then the model is fed 512 Tensors of shape
          [1, 256], one at a time.
        lr: the inner-loop learning rate for Reptile.

    Steps are taken using either cross entropy or binary
    cross entropy depending on the dtype of the outputs.

    Returns:
        A list of losses, where there is one loss for
          every step in the sequence. Each loss is an
          average over the entire mini-batch.
    """
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


def inferred_loss(out, targets):
    if targets.dtype.is_floating_point:
        return F.binary_cross_entropy_with_logits(out, targets)
    else:
        return F.cross_entropy(out.view(-1, *out.shape[2:]), targets.view(-1, *targets.shape[2:]))


def batched_run_sgd_epoch(model, inputs, outputs, lr):
    # Adjust for fact that we average over the
    # whole meta-batch.
    lr *= inputs.shape[0]

    device = next(model.parameters()).device
    parameters = model.batch_parameters(inputs.shape[0])
    losses = []
    for i in range(inputs.shape[1]):
        x = inputs[:, i:i+1].to(device)
        y = outputs[:, i:i+1].to(device)
        out = model.batch_forward(parameters, x)
        loss = inferred_loss(out, y)
        losses.append(loss.item())
        grads = torch.autograd.grad(loss, parameters)
        parameters = tuple((p - lr * g).detach().requires_grad_()
                           for p, g in zip(parameters, grads))
    return losses, parameters
