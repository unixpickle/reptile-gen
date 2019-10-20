import numpy as np
import torch
import torch.optim as optim

from reptile_gen.device import best_available_device
from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import batch_mnist_model
from reptile_gen.reptile import reptile_grad

from mnsit_train import OUT_PATH


def main():
    device = torch.device(best_available_device())
    model = batch_mnist_model()
    model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    history = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(train=False)):
        losses = reptile_grad(model, inputs, outputs, opt)
        history.append(np.mean(losses))
        print('step %d: loss=%f' % (i, np.mean(history)))


if __name__ == '__main__':
    main()
