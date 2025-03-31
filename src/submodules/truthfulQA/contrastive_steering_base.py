import contextlib
import functools
import gc
import io
import textwrap
from typing import Callable, List, Tuple
import argparse
import json
import einops
import numpy as np
import pandas as pd
import requests
import torch
from colorama import Fore
import os
from datasets import load_dataset
from jaxtyping import Float, Int
import time
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from DATA.dataset_truthfulqa import prepare_dataset
from UTILS.hook_utils import (
    activation_addition_cache_last_post,
    activation_ablation_cache_last_post,
    cache_activation_post_hook_pos,
    cache_activation_post_hook,
    add_hooks
)
from PCA.activation_pca import get_contrastive_activation_dim_reduction
from EVALUATE.evaluate_truthful_qa import evaluate_generation
from STAGE_STATS.stage_statistics import get_state_quantification

def generate_and_steer(
    cfg,
    model_base,
    input_ids,
    attention_mask,
    steering_vector=None,
    steering_method="addition",
    target_layer=None,
    position_to_cache=None,

):



    data_device = input_ids.device
    print(f"input_ids.device: {data_device}")
    data_device = attention_mask.device
    print(f"attention_mask.device: {data_device}")
    model_base.model = model_base.model.to(data_device)
    print(f"model_base.model.device: {model_base.model.device}")

    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    n_samples = len(input_ids)
    block_modules = model_base.model_block_modules
    len_prompt = input_ids.shape[1]

    # generation only, no steering, no caching
    if steering_method is None:
        cache = None
        fwd_hook = []
    if steering_method == "cache_pos":
        print(f"steering_method: {steering_method}")
        cache = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float16,
            device=model_base.model.device,
        )
        fwd_hook = [
            (
                block_modules[layer],
                cache_activation_post_hook_pos(
                    cache, layer, len_prompt=len_prompt, pos=-1  # last token
                ),
            )
            for layer in range(n_layers)
        ]
    if steering_method == "cache_all":

        cache = torch.zeros(
            (n_layers, n_samples,len_prompt, d_model),
            dtype=torch.float16,
            device=model_base.model.device,
        )
        # only cache the token position specified
 

        fwd_hook = [
            (
                block_modules[layer],
                cache_activation_post_hook(
                    cache, layer, len_prompt=len_prompt
                ),
            )
            for layer in range(n_layers)
        ]



    if steering_method == "addition":

        cache = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float16,
            device=model_base.model.device,
        )
        fwd_hook = [
            (
                block_modules[layer],
                activation_addition_cache_last_post(
                    steering_vector,
                    cache,
                    layer,
                    target_layer,
                    len_prompt=len_prompt,
                    pos=-1,  
                    steering_strength=cfg.steering_strength# last token
                ),
            )
            for layer in range(n_layers)
        ]

    elif steering_method == "ablation":

        cache = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float16,
            device=model_base.model.device,
        )
        fwd_hook = [
            (
                block_modules[layer],
                activation_ablation_cache_last_post(
                    steering_vector,
                    cache,
                    layer,
                    target_layer,
                    len_prompt=len_prompt,
                    pos=-1, 
                    steering_strength=cfg.steering_strength
                ),
            )
            for layer in range(n_layers)
        ]

    with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hook):
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
        )
        completions = model_base.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            generation_config=generation_config,
        )

    if position_to_cache is not None:
        # print(f"position_to_cache: {len(position_to_cache)}")
        cache_pos = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float16,
            device=model_base.model.device,
        )
        for i in range(n_samples):
            cache_pos[:, i, :] = cache[:,i,position_to_cache[i],:]
        # print(f"caching shape {cache_pos.shape}")
        return completions, cache_pos, len_prompt
    else:
        return completions, cache, len_prompt


def generate_and_steer_batch(cfg, model_base, dataloader, accelerator, 
                             steering_vector=None, steering_method=None):
    
    if torch.cuda.is_available():
        # Get the number of available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
    else:
        gpu_count = 0
        
    all_outputs = []
    all_cache = []
    start = time.time()

    for batch in dataloader:
        with torch.no_grad():
            # 1. Generate and steer for each batch and GPU
            if accelerator.is_main_process:
                print("generate and steer")
            completions, cache, len_input = generate_and_steer(
                            cfg,
                            model_base,
                            batch["input_ids"],
                            batch["attention_mask"],
                            steering_vector=steering_vector,
                            steering_method=steering_method,
                            target_layer=cfg.target_layer,
                            position_to_cache=None)

            outputs = model_base.tokenizer.batch_decode(completions[:, len_input:], 
                                                        skip_special_tokens=True)

            all_outputs.extend(outputs)
            if cache is not None:
                all_cache.append(cache)


    if len(all_cache) > 0:
        all_cache = torch.cat(all_cache, dim=1)


    end = time.time()

    if accelerator.is_main_process:
        print(f"Elapsed time: {end - start:.2f} seconds")
        print(f"len(all_completions before gather): {len(all_outputs)}")
    
    return all_outputs, all_cache

def save_activations(cfg, cache_positive, cache_negative,  
                     correct_choice,
                     model_choice_positive_all, model_choice_negative_all):
    # save activations
    activations = {}
    activations = {
        "activations_positive": cache_positive.float().detach().cpu(),
        "activations_negative": cache_negative.float().detach().cpu(),
        "correct_choice": correct_choice,
        "model_choice_negative":model_choice_negative_all,
        "model_choice_positive":model_choice_positive_all,
    }

    activation_path = os.path.join(cfg.artifact_path(), "activations")
    if not os.path.exists(activation_path):
        os.makedirs(activation_path)
    with open(
        activation_path
        + os.sep
        + f"{cfg.model_name}_activations_train.pkl",
        "wb",
    ) as f:
        pickle.dump(activations, f)
    print("activations saved ")#
    return activations

