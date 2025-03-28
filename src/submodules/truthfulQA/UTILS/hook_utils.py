import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from jaxtyping import Float, Int
from colorama import Fore

import contextlib
from typing import List, Tuple, Callable
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


def cache_activation_post_hook_pos(cache, layer, len_prompt, pos):
    """
    # USE DURING GENERATION
    :param cache: the cache of shape (n_layers, n_samples, d_model)
    :param layer: the layer to cache
    :param len_prompt: length of prompt
    :param pos: the token position to cache
    :return:
    """

    def hook_fn(module, input, output):
        nonlocal cache, layer, len_prompt, pos

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only cache the last token of the prompt not the generated answer
        if activation.shape[1] == len_prompt:
            cache[layer, :, :] = activation[:, pos, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def cache_activation_post_hook(cache, layer, len_prompt):
    """
    # USE DURING GENERATION
    :param cache: the cache of shape (n_layers, n_samples, len_prompt, d_model)
    :param layer: the layer to cache
    :param len_prompt: length of prompt
    :return:
    """

    def hook_fn(module, input, output):
        nonlocal cache, layer, len_prompt

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only cache the last token of the prompt not the generated answer
        if activation.shape[1] == len_prompt:
            cache[layer, :, :, :] = activation[:, :, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn



def activation_addition_cache_last_post(
    mean_diff,
    cache: Float[Tensor, "layer batch d_model"],
    layer,
    target_layer,
    len_prompt,
    pos,
    steering_strength=1
):
    """
    # USE DURING GENERATION

    """

    def hook_fn(module, input, output):
        nonlocal mean_diff, cache, layer, target_layer, len_prompt, pos, steering_strength

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        device = activation.device
        mean_diff = mean_diff.to(device)
        if layer == target_layer:
            # make sure the steering vector and the activaiton is in the same device for multiple gpus
            device = activation.device
            mean_diff = mean_diff.to(device)
            activation = activation + steering_strength * mean_diff
        
        # only cache the last token of the prompt not the generated answer
        if activation.shape[1] == len_prompt:
            cache[layer, :, :] = activation[:, pos, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def activation_ablation_cache_last_post(
    direction,
    cache: Float[Tensor, "layer batch d_model"],
    layer,
    target_layer,
    len_prompt,
    pos,
    steering_strength=1# last token

):
    """
    # USE DURING GENERATION

    """

    def hook_fn(module, input, output):
        nonlocal direction, cache, layer, target_layer, len_prompt, pos, steering_strength

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        if layer == target_layer:
            device = activation.device
            direction = direction.to(device)
            proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction

            activation = activation - proj
        # only cache the last token of the prompt not the generated answer
        if activation.shape[1] == len_prompt:
            cache[layer, :, :] = activation[:, pos, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn
