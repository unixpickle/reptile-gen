import os

import numpy as np
import torch
import torch.optim as optim

from reptile_gen.data import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_step

OUT_PATH = 'model.pt'
OPTIM_PATH = 'optimizer.pt'


def main():
    model = MNISTModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    last_100 = []
    opt = optim.Adam(model.parameters(), betas=(0, 0.999))
    if os.path.exists(OPTIM_PATH):
        opt.load_state_dict(torch.load(OPTIM_PATH))
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets()):
        losses = reptile_step(model, inputs, outputs, opt)
        loss = np.mean(losses)
        last_100.append(loss)
        last_100 = last_100[-100:]
        print('step %d: loss=%f last_100=%f' % (i, np.mean(losses), np.mean(last_100)))
        if i % 100 == 0:
            torch.save(model.state_dict(), OUT_PATH)
            torch.save(opt.state_dict(), OPTIM_PATH)


if __name__ == '__main__':
    main()
