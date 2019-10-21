import multiprocessing

import cloudpickle
import torch


def batched_grad(model, grad_fn, batch, threads=1, device='cpu'):
    def run_grad_fn(inputs, outputs):
        for p in model.parameters():
            p.grad = None
        d = torch.device(device)
        if device != 'cpu':
            model.to(d)
        res = grad_fn(model, inputs.to(d), outputs.to(d))
        return [p.grad.cpu() for p in model.parameters()], res

    pickled_fn = cloudpickle.dumps(run_grad_fn)
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(min(len(batch), threads)) as pool:
        raw_results = pool.map(call_pickled_fn, [(pickled_fn, x) for x in batch])
    grads, results = list(zip(*raw_results))

    for grad in grads:
        for p, g in zip(model.parameters(), grad):
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)

    return list(results)


def call_pickled_fn(data_args):
    data, args = data_args
    res = cloudpickle.loads(data)(*args)
    return res
