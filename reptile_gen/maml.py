import torch
import torch.nn.functional as F


def maml_grad(model, inputs, outputs, lr, batch=1):
    """
    Update the model gradient using MAML.
    """
    params = list(model.parameters())
    device = params[0].device
    initial_values = []
    final_values = []
    loss_grads = []
    scalar_losses = []
    for i in range(0, inputs.shape[0], batch):
        x = inputs[i:i+batch]
        y = outputs[i:i+batch]
        target = y.to(device)
        out = model(x.to(device))
        if target.dtype.is_floating_point:
            loss = F.binary_cross_entropy_with_logits(out, target)
        else:
            loss = F.cross_entropy(out, target)
        scalar_losses.append(loss.item())
        initial_values.append([p.clone().detach() for p in params])
        updated = []
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        loss_grads.append([g.detach() for g in grads])
        for grad, param in zip(grads, params):
            x = param - lr * grad
            updated.append(x)
            param.data.copy_(x)
        final_values.append(updated)
    gradient = [torch.zeros_like(p) for p in params]
    for loss_grad, initial, final in list(zip(loss_grads, initial_values, final_values))[::-1]:
        for p, x in zip(params, initial):
            p.data.copy_(x)
        future_grad = torch.autograd.grad(final, params, grad_outputs=gradient, retain_graph=True)
        gradient = [v1 + v2 for v1, v2 in zip(loss_grad, future_grad)]
    for p, g in zip(params, gradient):
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)
    return scalar_losses
