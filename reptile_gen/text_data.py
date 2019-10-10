import random

import numpy as np
import torch


def iterate_mini_datasets(path, max_len=128):
    for line in iterate_lines(path):
        if len(line) >= max_len - 1:
            line = line[:max_len - 1]
        inputs = []
        outputs = []
        for i, ch in enumerate(line + [0]):
            inputs.append([i])
            outputs.append(ch)
        yield (torch.from_numpy(np.array(inputs)).long(),
               torch.from_numpy(np.array(outputs)).long())


def iterate_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    while True:
        random.shuffle(lines)
        for l in lines:
            yield [int(x) for x in bytes(l, 'utf-8')]
