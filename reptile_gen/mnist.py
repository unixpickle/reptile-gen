import random

import numpy as np
import torch
from torchvision import datasets, transforms


def iterate_mini_datasets(train=True, ordered=False):
    for image in iterate_images(train=train):
        image = image.flatten()
        inputs = []
        outputs = []
        for i in pixel_indices(ordered=ordered):
            inputs.append([i // 28, i % 28])
            outputs.append([0.0 if image[i] < 0.5 else 1.0])
        yield (torch.from_numpy(np.array(inputs)).long(),
               torch.from_numpy(np.array(outputs)).float())


def pixel_indices(ordered=False):
    res = list(range(28 * 28))
    if ordered:
        return res
    return random.sample(res, len(res))


def iterate_images(train=True):
    mnist = datasets.MNIST('data', train=train, download=True,
                           transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)
    while True:
        for images, _ in loader:
            yield images[0].numpy()
