import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from reptile_gen.model import TextModel

from text_train import OUT_PATH


def main():
    model = TextModel()
    model.load_state_dict(torch.load(OUT_PATH))
    opt = optim.SGD(model.parameters(), lr=1e-3)

    sequence = []

    for i in range(128):
        inputs = torch.from_numpy(np.array([[i]])).long()
        logits = model(inputs)
        probs = F.softmax(logits[0], dim=0).detach().cpu().numpy()
        sample = np.random.choice(np.arange(256), p=probs)
        sequence.append(int(sample))
        if sample == 0:
            break
        targets = torch.from_numpy(np.array([sample])).long()
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(str(bytes([min(0x79, x) for x in sequence]), 'ascii'))


if __name__ == '__main__':
    main()
