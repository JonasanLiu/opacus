from functorch import make_functional, vmap, grad


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


def compute_per_sample_gradient(layer, activations, backprops):
    per_sample_grads = layer.ft_compute_sample_grad(layer.func_params, activations, backprops)

    ret = {}
    for i_p, p in enumerate(layer.parameters()):
        ret[p] = per_sample_grads[i_p]

    return ret
