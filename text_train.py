import itertools
import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.device import best_available_device
from reptile_gen.model import batch_text_model
from reptile_gen.reptile import batched_reptile_grad
from reptile_gen.text_data import iterate_mini_datasets

OUT_PATH = 'text_model.pt'
DATASET = 'text_data/dir_listing.txt'
AVG_SIZE = 20
META_BATCH = 50
INNER_LR = 1e-3


def main():
    device = torch.device(best_available_device())
    model = batch_text_model()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    outer_opt = optim.Adam(model.parameters(), lr=1e-3)
    mini_batches = iterate_mini_datasets(DATASET)

    last_n = []
    for i in itertools.count():
        batch = [next(mini_batches) for _ in range(META_BATCH)]
        outer_opt.zero_grad()
        losses = batched_reptile_grad(model, batch, INNER_LR)
        outer_opt.step()
        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        model.cpu()
        torch.save(model.state_dict(), OUT_PATH)
        model.to(device)
        print('step %d: loss=%f last_%d=%f' % (i, loss, AVG_SIZE, np.mean(last_n)))


if __name__ == '__main__':
    main()
