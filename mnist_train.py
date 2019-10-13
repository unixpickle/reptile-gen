import itertools
import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_grad

OUT_PATH = 'model.pt'
AVG_SIZE = 1000
META_BATCH = 50


def main():
    model = MNISTModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    outer_opt = optim.Adam(model.parameters(), lr=1e-3)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    mini_batches = iterate_mini_datasets()
    last_n = []
    for i in itertools.count():
        inputs, outputs = next(mini_batches)
        losses = reptile_grad(model, inputs, outputs, opt)
        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        if i % META_BATCH == 0:
            outer_opt.step()
            outer_opt.zero_grad()
            torch.save(model.state_dict(), OUT_PATH)
            print('step %d: loss=%f last_%d=%f' %
                  (i//META_BATCH, np.mean(losses), AVG_SIZE, np.mean(last_n)))


if __name__ == '__main__':
    main()
