import numpy as np
import torch

from .model import make_mnist_model


def test_batched_models():
    m = make_mnist_model()
    m.to(torch.float64)
    inputs = torch.from_numpy(np.array([[1, 2], [3, 4], [5, 1]])).long()[:, None]
    batch_out = m.batch_forward(m.batch_parameters(3), inputs).detach().numpy()
    regular_out = torch.stack([m(inputs[i]) for i in range(3)]).detach().numpy()
    assert np.allclose(batch_out, regular_out)
