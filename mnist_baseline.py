import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reptile_gen.device import best_available_device
from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import MNISTBaseline

OUT_PATH = 'model_baseline.pt'
AVG_SIZE = 20
BATCH = 50


def main():
    device = torch.device(best_available_device())
    model = MNISTBaseline()
    if os.path.exists(OUT_PATH):
        model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    samples = iterate_mini_datasets()
    last_n = []
    for i in itertools.count():
        input_batch = []
        output_batch = []
        for inputs, outputs in [next(samples) for _ in range(BATCH)]:
            shifted_outputs = torch.cat([torch.zeros_like(outputs[:1]), outputs[:-1]], dim=0)
            ins = torch.cat([inputs, shifted_outputs.long()], dim=-1)
            input_batch.append(ins)
            output_batch.append(outputs)
        inputs = torch.stack(input_batch, dim=1).to(device)
        outputs = torch.stack(output_batch, dim=1).to(device)
        logits, _ = model(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, outputs)
        last_n.append(loss.item())
        last_n = last_n[-AVG_SIZE:]

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.to(torch.device('cpu'))
        torch.save(model.state_dict(), OUT_PATH)
        model.to(device)

        print('step %d: loss=%f last_%d=%f' % (i, loss.item(), AVG_SIZE, np.mean(last_n)))


if __name__ == '__main__':
    main()
