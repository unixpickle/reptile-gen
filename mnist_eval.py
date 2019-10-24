import numpy as np
import torch

from reptile_gen.device import best_available_device
from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import make_mnist_model
from reptile_gen.reptile import reptile_grad

from mnist_train import OUT_PATH, INNER_LR


def main():
    device = torch.device(best_available_device())
    model = make_mnist_model()
    model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    history = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(train=False)):
        losses = reptile_grad(model, [(inputs, outputs)], INNER_LR)
        history.append(np.mean(losses))
        print('step %d: loss=%f' % (i, np.mean(history)))


if __name__ == '__main__':
    main()
