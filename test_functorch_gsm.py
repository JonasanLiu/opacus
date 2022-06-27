from functorch import make_functional, vmap, grad
from torch import nn
from opacus import GradSampleModule
import torch

import torch.nn.functional as F

from opacus.grad_sample.utils import register_grad_sampler

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def prepare_layer(layer):
    flayer, params = make_functional(layer)
    def compute_loss_stateless_model(params, activations, backprops):
        batched_activations = activations.unsqueeze(0)
        batched_backprops = backprops.unsqueeze(0)

        output = flayer(params, batched_activations)
        loss = (output * batched_backprops).sum()

        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)

    layer.ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
    layer.func_params = params



print("Making mylinear")
layer = MyLinear(10, 10)
print("prepare")
prepare_layer(layer)
print("done")


@register_grad_sampler(MyLinear)
def compute_per_sample_gradient(layer, activations, backprops):
    print("HEYHEY")
    per_sample_grads = layer.ft_compute_sample_grad(layer.func_params, activations, backprops)

    ret = {}
    ret[layer.weight] = per_sample_grads[0]
    if len(per_sample_grads) > 1:
        ret[layer.bias] = per_sample_grads[1]

    return ret


module = GradSampleModule(layer)

X = torch.randn(32, 10)
Y = layer(X)
loss = (Y*Y).sum()
loss.backward()

print("ALLCLOSE", torch.allclose(layer.weight.grad_sample.mean(0), layer.weight.grad))