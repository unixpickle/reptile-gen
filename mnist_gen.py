import random

from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from reptile_gen.device import best_available_device
from reptile_gen.mnist import pixel_indices
from reptile_gen.model import MNISTModel
from mnist_train import OUT_PATH


def main():
    device = torch.device(best_available_device())

    model = MNISTModel()
    model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=1e-3)

    output = np.zeros([28 * 28], dtype=np.uint8)
    for i in pixel_indices():
        inputs = torch.from_numpy(np.array([i // 28, i % 28])).to(device).long()
        outs = model(inputs[None])[0]
        out_prob = torch.sigmoid(outs).item()
        if random.random() < out_prob:
            output[i] = 255
            target = [1.0]
        else:
            output[i] = 0
            target = [0.0]
        outs = model(inputs[None])[0]
        float_target = torch.from_numpy(np.array(target)).to(device).float()
        loss = F.binary_cross_entropy_with_logits(outs, float_target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    output = output.reshape([28, 28, 1])
    output = np.concatenate([output]*3, axis=-1)
    Image.fromarray(output).save('sample.png')


if __name__ == '__main__':
    main()
