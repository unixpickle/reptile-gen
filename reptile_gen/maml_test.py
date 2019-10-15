import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from maml import maml_grad


def test_maml_grad():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 3),
        nn.Tanh(),
        nn.Linear(3, 1),
    )
    model.to(torch.double)
    inputs = torch.from_numpy(np.array([[1.0, 2.0, -0.5], [0.5, 1, -1]]))
    outputs = torch.from_numpy(np.array([[0.9], [0.2]]))

    # Make sure single-step gradients are correct
    # without any numerical approximation.
    exact_grads = _exact_maml_grad(model, inputs[:1], outputs[:1], 0.01)
    step_grads = _numerical_maml_grad(model, inputs[:1], outputs[:1], 0.01)
    for i, (ex, ap) in enumerate(zip(exact_grads, step_grads)):
        assert np.allclose(ex, ap, rtol=1e-4, atol=1e-4)

    exact_grads = _exact_maml_grad(model, inputs, outputs, 0.01)
    approx_grads = _numerical_maml_grad(model, inputs, outputs, 0.01)
    for ex, ap in zip(exact_grads, approx_grads):
        assert np.allclose(ex, ap, rtol=1e-4, atol=1e-4)


def _exact_maml_grad(model, inputs, outputs, lr):
    for p in model.parameters():
        p.grad = None
    maml_grad(model, inputs, outputs, lr)
    return [p.grad.numpy() for p in model.parameters()]


def _one_step_maml_grad(model, inputs, outputs, lr):
    for p in model.parameters():
        p.grad = None
    F.binary_cross_entropy_with_logits(model(inputs), outputs).backward()
    return [p.grad.numpy() for p in model.parameters()]


def _numerical_maml_grad(model, inputs, outputs, lr, delta=1e-4):
    grad = []
    for p in model.parameters():
        param_grad = []
        np_value = p.detach().numpy()
        flat_np = np_value.reshape([-1])
        for i, x in enumerate(flat_np):
            flat_np[i] = x - delta
            p.data.copy_(torch.from_numpy(np_value).to(p.device))
            loss1 = _numerical_maml_loss(model, inputs, outputs, lr)

            flat_np[i] = x + delta
            p.data.copy_(torch.from_numpy(np_value).to(p.device))
            loss2 = _numerical_maml_loss(model, inputs, outputs, lr)

            flat_np[i] = x
            p.data.copy_(torch.from_numpy(np_value).to(p.device))

            param_grad.append((loss2 - loss1) / (2 * delta))
        grad.append(np.array(param_grad, dtype=np.float64).reshape(p.shape))
    return grad


def _numerical_maml_loss(model, inputs, outputs, lr):
    backup = [p.data.clone() for p in model.parameters()]
    opt = optim.SGD(model.parameters(), lr)
    losses = []
    for i in range(inputs.shape[0]):
        x, y = inputs[i:i+1], outputs[i:i+1]
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    for p, b in zip(model.parameters(), backup):
        p.data.copy_(b)
    return np.sum(losses)
