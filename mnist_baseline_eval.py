import itertools

import numpy as np
import torch
import torch.nn.functional as F

from reptile_gen.device import best_available_device
from reptile_gen.mnist import iterate_mini_datasets
from reptile_gen.model import MNISTBaseline
from mnist_baseline import OUT_PATH, BATCH


def main():
    device = torch.device(best_available_device())
    model = MNISTBaseline()
    model.load_state_dict(torch.load(OUT_PATH))
    model.to(device)

    samples = iterate_mini_datasets(train=False)
    history = []
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
        history.append(loss.item())
        print('samples=%d loss=%f' % (i * BATCH, np.mean(history)))


if __name__ == '__main__':
    main()
