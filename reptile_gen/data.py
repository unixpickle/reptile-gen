import random

import numpy as np
import torch
from torchvision import datasets, transforms


def iterate_mini_datasets(train=True):
    for image in iterate_images(train=train):
        image = image.flatten()
        inputs = []
        outputs = []
        for i in random.sample(list(range(len(image))), len(image)):
            inputs.append([i // 28, i % 28])
            outputs.append([0.0 if image[i] < 0.5 else 1.0])
        yield np.array(inputs, dtype='long'), np.array(outputs, dtype='float32')


def iterate_images(train=True):
    mnist = datasets.MNIST('data', train=train, download=True,
                           transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(mnist, batch_size=1, shuffle=True)
    while True:
        try:
            for images, _ in loader:
                yield images[0].numpy()
        except StopIteration:
            pass
