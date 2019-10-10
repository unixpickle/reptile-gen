import itertools
import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_grad

OUT_PATH = 'model.pt'
OPTIM_PATH = 'optimizer.pt'
AVG_SIZE = 1000
BIG_BATCH = 28
BIG_ITERS = 19


def main():
    model = MNISTModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    last_n = []
    outer_opt = optim.SGD(model.parameters(), lr=0.01)
    opt = optim.Adam(model.parameters(), lr=2e-4, betas=(0, 0.999))
    if os.path.exists(OPTIM_PATH):
        opt.load_state_dict(torch.load(OPTIM_PATH))
    big_opt = optim.Adam(model.parameters(), betas=(0, 0.999))
    mini_batches = iterate_mini_datasets()
    for i in itertools.count():
        big_losses = []
        for _ in range(BIG_ITERS):
            inputs, outputs = next(mini_batches)
            losses = reptile_grad(model, inputs, outputs, big_opt, batch=BIG_BATCH)
            outer_opt.step()
            big_losses.append(losses)
        inputs, outputs = next(mini_batches)
        losses = reptile_grad(model, inputs, outputs, opt)
        outer_opt.step()
        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        print('step %d: loss=%f big_loss=%f last_%d=%f' %
              (i, np.mean(losses), np.mean(big_losses), AVG_SIZE, np.mean(last_n)))
        if i % 100 == 0:
            torch.save(model.state_dict(), OUT_PATH)
            torch.save(opt.state_dict(), OPTIM_PATH)


if __name__ == '__main__':
    main()
