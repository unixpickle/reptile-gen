import numpy as np
import torch
import torch.optim as optim

from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_grad

from mnsit_train import OUT_PATH, INNER_ITERS


def main():
    model = MNISTModel()
    model.load_state_dict(torch.load(OUT_PATH))
    opt = optim.SGD(model.parameters(), lr=1e-3)
    history = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(train=False)):
        losses = reptile_grad(model, inputs, outputs, opt, inner_iters=INNER_ITERS)
        history.append(np.mean(losses))
        print('step %d: loss=%f' % (i, np.mean(history)))


if __name__ == '__main__':
    main()
