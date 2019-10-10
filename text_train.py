import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.model import TextModel
from reptile_gen.reptile import reptile_grad
from reptile_gen.text_data import iterate_mini_datasets

OUT_PATH = 'model.pt'
OPTIM_PATH = 'optimizer.pt'
DATASET = 'text_data/dir_listing.txt'
AVG_SIZE = 1000


def main():
    model = TextModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    last_n = []
    outer_opt = optim.Adam(model.parameters())
    opt = optim.Adam(model.parameters(), lr=2e-4, betas=(0, 0.999))
    if os.path.exists(OPTIM_PATH):
        opt.load_state_dict(torch.load(OPTIM_PATH))
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(DATASET)):
        losses = reptile_grad(model, inputs, outputs, opt)
        outer_opt.step()
        loss = np.mean(losses)
        last_n.append(loss)
        last_n = last_n[-AVG_SIZE:]
        print('step %d: loss=%f last_%d=%f' %
              (i, np.mean(losses), AVG_SIZE, np.mean(last_n)))
        if i % 10 == 0:
            torch.save(model.state_dict(), OUT_PATH)
            torch.save(opt.state_dict(), OPTIM_PATH)


if __name__ == '__main__':
    main()
