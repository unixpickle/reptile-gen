import random

from PIL import Image
import numpy as np
import torch

from reptile_gen.device import best_available_device
from reptile_gen.mnist import pixel_indices
from reptile_gen.model import MNISTBaseline

from mnist_baseline import OUT_PATH

GRID_SIZE = 4


def main():
    device = torch.device(best_available_device())
    model = MNISTBaseline()
    model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)

    grid = np.zeros([28 * GRID_SIZE, 28 * GRID_SIZE, 3], dtype=np.uint8)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            print('generating tile %d,%d' % (i, j))
            grid[28*i:28*(i+1), 28*j:28*(j+1)] = generate_single(model, device)

    Image.fromarray(grid).save('samples.png')


def generate_single(model, device):
    output = np.zeros([28 * 28], dtype=np.uint8)
    previous = 0
    hidden = None
    for i in pixel_indices():
        inputs = torch.from_numpy(np.array([i // 28, i % 28, previous])).to(device).long()
        logits, hidden = model(inputs[None, None], hidden=hidden)
        out_prob = torch.sigmoid(logits).item()
        previous = (1 if random.random() < out_prob else 0)
        output[i] = previous * 255
    output = output.reshape([28, 28, 1])
    output = np.concatenate([output]*3, axis=-1)
    return output


if __name__ == '__main__':
    main()
