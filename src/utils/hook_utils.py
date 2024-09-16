import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import numpy as np


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def patch_residual_hook_pre(
    clean_cache: Float[Tensor, "layer batch pos d_model"],
    patch_layer: int,
    layer: int,
    patch_pos: int,
):
    def hook_fn(module, input):
        # nonlocal cache, layer, positions, batch_id, batch_size, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        if patch_layer == layer:
            activation[:, patch_pos, :] = clean_cache[patch_layer, :, patch_pos, :]

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def patch_residual_hook_post(
    clean_cache: Float[Tensor, "layer batch pos d_model"],
    patch_layer: int,
    layer: int,
    patch_pos: int,
):
    def hook_fn(module, input, output):

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        if patch_layer == layer:
            activation[:, patch_pos, :] = clean_cache[patch_layer, :, patch_pos, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def cache_residual_hook_post(
    cache: Float[Tensor, "layer batch seq_len d_model"],
    layer: int,
):
    def hook_fn(module, input, output):

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch seq_len d_model"] = output

        cache[layer, :, :, :] = activation[:, :, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def cache_residual_hook_pre(
    cache: Float[Tensor, "layer batch seq_len d_model"],
    layer: int,
):
    def hook_fn(module, input):

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch seq_len d_model"] = input

        cache[layer, :, :, :] = activation[:, :, :]

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn
