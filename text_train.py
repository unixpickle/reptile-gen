import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.model import TextModel
from reptile_gen.reptile import reptile_grad
from reptile_gen.text_data import iterate_mini_datasets

OUT_PATH = 'text_model.pt'
DATASET = 'text_data/dir_listing.txt'
AVG_SIZE = 1000
META_BATCH = 50


def main():
    model = TextModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    last_n = []
    outer_opt = optim.Adam(model.parameters(), lr=1e-3)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    outer_opt.zero_grad()
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(DATASET)):
        losses = reptile_grad(model, inputs, outputs, opt)
        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        if i % META_BATCH == 0:
            print('step %d: loss=%f last_%d=%f' %
                  (i//META_BATCH, np.mean(losses), AVG_SIZE, np.mean(last_n)))
            outer_opt.step()
            outer_opt.zero_grad()
            torch.save(model.state_dict(), OUT_PATH)


if __name__ == '__main__':
    main()
