import random

from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from reptile_gen.model import MNISTModel

IN_PATH = 'model.pt'


def main():
    model = MNISTModel()
    model.load_state_dict(torch.load(IN_PATH))

    opt = optim.SGD(model.parameters(), lr=1e-3)
    output = np.zeros([28 * 28], dtype=np.uint8)
    for i in random.sample(list(range(len(output))), len(output)):
        inputs = torch.from_numpy(np.array([i // 28, i % 28])).long()
        outs = model(inputs[None])[0]
        out_prob = torch.sigmoid(outs).item()
        if random.random() < out_prob:
            output[i] = 255
            target = [1.0]
        else:
            output[i] = 0
            target = [0.0]
        loss = F.binary_cross_entropy_with_logits(outs, torch.from_numpy(np.array(target)).float())
        opt.zero_grad()
        loss.backward()
        opt.step()

    output = output.reshape([28, 28, 1])
    output = np.concatenate([output]*3, axis=-1)
    Image.fromarray(output).save('sample.png')


if __name__ == '__main__':
    main()
