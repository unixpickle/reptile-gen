import os

import numpy as np
import torch

from reptile_gen.data import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_step

OUT_PATH = 'model.pt'


def main():
    model = MNISTModel()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    last_100 = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets()):
        losses = reptile_step(model, inputs, outputs)
        loss = np.mean(losses)
        last_100.append(loss)
        last_100 = last_100[-100:]
        print('step %d: loss=%f last_100=%f' % (i, np.mean(losses), np.mean(last_100)))
        if i % 100 == 0:
            torch.save(OUT_PATH, model.state_dict())


if __name__ == '__main__':
    main()
