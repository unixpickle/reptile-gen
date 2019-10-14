from multiprocessing import Pool, set_start_method

import cloudpickle
import torch


def batched_grad(model, grad_fn, batch, threads=1, device='cpu'):
    set_start_method('spawn', force=True)

    model_class = model.__class__
    model_dict = {x: y.cpu().numpy() for x, y in model.state_dict().items()}

    def run_grad_fn(inputs, outputs):
        model = model_class()
        state = {x: torch.from_numpy(y) for x, y in model_dict.items()}
        model.load_state_dict(state)
        d = torch.device(device)
        if device != 'cpu':
            model.to(d)
        res = grad_fn(model, inputs.to(d), outputs.to(d))
        return [p.grad for p in model.parameters()], res

    pool = Pool(min(len(batch), threads))
    pickled_fn = cloudpickle.dumps(run_grad_fn)
    raw_results = pool.map(call_pickled_fn, [(pickled_fn, x) for x in batch])
    grads, results = list(zip(*raw_results))
    pool.close()

    for grad in grads:
        for p, g in zip(model.parameters(), grad):
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)

    return results


def call_pickled_fn(data_args):
    data, args = data_args
    res = cloudpickle.loads(data)(*args)
    return res
