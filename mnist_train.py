import itertools
import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.device import best_available_device
from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import make_mnist_model
from reptile_gen.reptile import reptile_grad

OUT_PATH = 'model.pt'
AVG_SIZE = 20
META_BATCH = 50
INNER_LR = 1e-3


def main():
    device = torch.device(best_available_device())
    model = make_mnist_model()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    outer_opt = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    mini_batches = iterate_mini_datasets()
    last_n = []
    for i in itertools.count():
        batch = [next(mini_batches) for _ in range(META_BATCH)]

        outer_opt.zero_grad()
        losses = reptile_grad(model, batch, INNER_LR)
        outer_opt.step()

        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        model.cpu()
        torch.save(model.state_dict(), OUT_PATH)
        model.to(device)
        print('step %d: loss=%f last_%d=%f' % (i, np.mean(losses), AVG_SIZE, np.mean(last_n)))


if __name__ == '__main__':
    main()
