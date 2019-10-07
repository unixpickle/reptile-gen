import numpy as np

from reptile_gen.data import iterate_mini_datasets
from reptile_gen.model import MNISTModel
from reptile_gen.reptile import reptile_step


def main():
    model = MNISTModel()
    last_100 = []
    for i, (inputs, outputs) in enumerate(iterate_mini_datasets()):
        losses = reptile_step(model, inputs, outputs)
        loss = np.mean(losses)
        last_100.append(loss)
        last_100 = last_100[-100:]
        print('step %d: loss=%f last_100=%f' % (i, np.mean(losses), np.mean(last_100)))


if __name__ == '__main__':
    main()
