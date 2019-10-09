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
    model.load_state_dict(torch.load(OUT_PATH))
    opt = optim.Adam(model.parameters(), betas=(0, 0.999))
    opt.load_state_dict(torch.load(OPTIM_PATH))
    history = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets(train=False)):
        losses = reptile_step(model, inputs, outputs, opt)
        history.append(np.mean(losses))
        print('step %d: loss=%f' % (i, np.mean(history)))


if __name__ == '__main__':
    main()
