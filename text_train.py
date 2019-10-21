import itertools
import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.batching import batched_grad
from reptile_gen.device import best_available_device
from reptile_gen.model import batch_text_model
from reptile_gen.reptile import reptile_grad
from reptile_gen.text_data import iterate_mini_datasets

OUT_PATH = 'text_model.pt'
DATASET = 'text_data/dir_listing.txt'
AVG_SIZE = 20
META_BATCH = 50
INNER_LR = 1e-3


def main():
    model = batch_text_model()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    outer_opt = optim.Adam(model.parameters(), lr=1e-3)
    opt = optim.SGD(model.parameters(), lr=INNER_LR)
    mini_batches = iterate_mini_datasets(DATASET)

    last_n = []
    for i in itertools.count():
        outer_opt.zero_grad()

        def grad_fn(model, x, y):
            return reptile_grad(model, x, y, opt)

        batch = [next(mini_batches) for _ in range(META_BATCH)]
        threads = 1
        if best_available_device() != 'cpu':
            threads = 4
        losses = batched_grad(model, grad_fn, batch,
                              device=best_available_device(),
                              threads=threads)
        loss = np.mean([np.mean(x) for x in losses])
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        outer_opt.step()
        torch.save(model.state_dict(), OUT_PATH)
        print('step %d: loss=%f last_%d=%f' % (i, loss, AVG_SIZE, np.mean(last_n)))


if __name__ == '__main__':
    main()
