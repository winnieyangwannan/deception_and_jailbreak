import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
from fancy_einsum import einsum
import gc
import os
from typing_extensions import Literal
from tqdm import tqdm
from functools import partial

import argparse
from pipeline.configs.config_generation import Config
from datasets import load_dataset
import sae_lens
import pysvelte
from torchtyping import TensorType as TT
from IPython import get_ipython
from pathlib import Path
import pickle
import numpy as np
from matplotlib.widgets import EllipseSelector
from IPython.display import HTML, Markdown
import torch
from src.submodules.dataset.task_and_dataset_deception import (
    ContrastiveDatasetDeception,
    tokenize_prompts_and_answers,
)
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import line, scatter, bar
from src.utils.neel_plotly_utils import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern
from src.submodules.evaluations.evaluate_deception import (
    evaluate_generation_deception,
    plot_lying_honest_performance,
)


class ContrastiveModelBase:
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset
        self.contrastive_type = cfg.contrastive_type
        self.answer_type = cfg.answer_type

        # tokenize prompts and answer
        (
            self.prompt_tokens_positive,
            self.answer_tokens_positive_correct,
            self.answer_tokens_positive_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            self.dataset.prompts_positive,
            self.dataset.answers_positive_correct,
            self.dataset.answers_positive_wrong,
        )
        (
            self.prompt_tokens_negative,
            self.answer_tokens_negative_correct,
            self.answer_tokens_negative_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            self.dataset.prompts_negative,
            self.dataset.answers_negative_correct,
            self.dataset.answers_negative_wrong,
        )

        self.positive_cache = None
        self.negative_cache = None

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")
        print("test positive prompt:")
        utils.test_prompt(
            self.dataset.prompts_positive[0],
            self.dataset.answers_positive_correct[0],
            self.model,
            prepend_bos=True,
        )

        print("test negative prompt:")
        utils.test_prompt(
            self.dataset.prompts_negative[0],
            self.dataset.answers_negative_correct[0],
            self.model,
            prepend_bos=True,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

        # save the visualization
        save_path = self.cfg.artifact_path()
        self.vis_path = save_path + os.sep + "attribution_patching"
        if not os.path.exists(self.vis_path):
            os.makedirs(self.vis_path)

        print("Memory after initialization:")
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def run_contrastive_activation_cache(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """

        positive_cache = self.run_and_cache_batch(self.prompt_tokens_positive)

        negative_cache = self.run_and_cache_batch(self.prompt_tokens_negative)

        self.positive_cache = positive_cache

        self.negative_cache = negative_cache
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def generate_and_contrastive_activation_cache(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """

        positive_cache = self.generate_and_cache_batch(
            self.dataset.prompts_positive,
            self.prompt_tokens_positive,
            self.cfg.contrastive_type[0],
        )

        negative_cache = self.generate_and_cache_batch(
            self.dataset.prompts_negative,
            self.prompt_tokens_negative,
            self.cfg.contrastive_type[1],
        )

        self.positive_cache = positive_cache

        self.negative_cache = negative_cache
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def get_contrastive_activation_pca(self, pos=-1):

        # 1. Get access to cache
        activations_positive = [
            self.positive_cache[utils.get_act_name("resid_post", l)].squeeze()
            for l in range(self.model.cfg.n_layers)
        ]

        activations_negative = [
            self.negative_cache[utils.get_act_name("resid_post", l)].squeeze()
            for l in range(self.model.cfg.n_layers)
        ]

        # stack the list of activations
        activations_positive = torch.stack(
            activations_positive
        )  # shape = [batch,d_model]
        activations_negative = torch.stack(activations_negative)

        # 2. Get pca and plot
        fig = get_pca_and_plot(
            self.cfg, activations_positive, activations_negative, self.dataset
        )

        # 3. Save plot
        pca_path = os.path.join(self.cfg.artifact_path(), "pca")
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(pca_path + os.sep + f"{self.cfg.model_alias}_pca.html")
        pio.write_image(
            fig, pca_path + os.sep + f"{self.cfg.model_alias}_pca.png", scale=6
        )

    def run_and_cache_batch(self, prompt_tokens):

        cache_all = self.initialize_cache(prompt_tokens)
        n_prompts = len(prompt_tokens)
        for batch in tqdm(range(1, n_prompts, self.cfg.batch_size)):

            cache = run_and_cache(
                self.model,
                prompt_tokens[batch : batch + self.cfg.batch_size],
            )
            for key in cache.keys():
                cache_all.cache_dict[key] = torch.cat(
                    (cache_all[key], cache[key]), dim=0
                )

        del cache
        gc.collect()
        torch.cuda.empty_cache()
        return cache_all

    def generate_and_cache_batch(self, prompts, prompt_tokens, contrastive_type):

        # 1. Gen data
        statements = self.dataset.statements
        labels = self.dataset.answers_positive_correct

        # 2. Generation intialization
        completions = []
        output, cache_all = self.initialize_cache(
            prompt_tokens, mode="generate", max_new_tokens=self.cfg.max_new_tokens
        )
        # Construct generation result for saving (one batch)
        generation_toks = output[:, prompt_tokens.shape[1] :]
        completions.append(
            {
                "prompt": prompts[0],
                "statement": statements[0],
                "response": self.model.tokenizer.decode(
                    generation_toks[0], skip_special_tokens=True
                ).strip(),
                "label": labels[0],
                "ID": 0,
            }
        )
        # 3. Main loop
        n_prompts = len(prompt_tokens)
        for batch in tqdm(range(1, n_prompts, self.cfg.batch_size)):

            output, cache = generate_and_cache(
                self.model,
                prompt_tokens[batch : batch + self.cfg.batch_size],
                max_new_tokens=self.cfg.max_new_tokens,
            )

            # concatenate cache
            for key in cache.keys():
                cache_all.cache_dict[key] = torch.cat(
                    (cache_all[key], cache[key]), dim=0
                )

            # Construct generation result for saving (one batch)
            generation_toks = output[:, prompt_tokens.shape[1] :]
            for generation_idx, generation in enumerate(generation_toks):
                completions.append(
                    {
                        "statement": statements[batch + generation_idx],
                        "response": self.model.tokenizer.decode(
                            generation, skip_special_tokens=True
                        ).strip(),
                        "label": labels[batch + generation_idx],
                        "ID": batch + generation_idx,
                    }
                )

        # 3. Save completion
        completion_path = os.path.join(self.cfg.artifact_path(), "completions")
        if not os.path.exists(completion_path):
            os.makedirs(completion_path)
        with open(
            completion_path
            + os.sep
            + f"{self.cfg.model_alias}_completions_{contrastive_type}.json",
            "w",
        ) as f:
            json.dump(completions, f, indent=4)

        return cache_all

    def initialize_cache(self, prompt_tokens, mode="run", max_new_tokens=100):

        # initialize cache
        if mode == "run":
            cache = run_and_cache(
                self.model,
                prompt_tokens[0 : 0 + 1],
            )
            return cache
        else:
            output, cache = generate_and_cache(
                self.model,
                prompt_tokens[0 : 0 + 1],
                max_new_tokens=max_new_tokens,
                do_sample=self.cfg.do_sample,
            )

            return output, cache

    def evaluate_deception(self):
        completions_honest = evaluate_generation_deception(
            self.cfg, contrastive_type=self.cfg.contrastive_type[0]
        )
        completions_lying = evaluate_generation_deception(
            self.cfg, contrastive_type=self.cfg.contrastive_type[1]
        )
        plot_lying_honest_performance(self.cfg, completions_honest, completions_lying)


def run_with_cache_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits, positive_cache = model.run_with_cache(positive_tokens)
    negative_logits, negative_cache = model.run_with_cache(negative_tokens)

    return positive_logits, negative_logits, positive_cache, negative_cache


def run_model_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits = model(positive_tokens)
    negative_logits = model(negative_tokens)

    return positive_logits, negative_logits


def run_and_cache(model, tokens):
    # Always reset hook before you start -- good habit!
    model.reset_hooks()

    # (1) Create filter for what to cache
    activation_filter = lambda name: "resid_post" in name

    # (2) Construct hook functions

    # forward cache hook: cache the activation during forward run
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act[:, -1, :].detach()

    # (3) Construct hooks
    model.add_hook(activation_filter, forward_cache_hook, "fwd")

    # (4) Run forward pass
    model(tokens)

    # Always reset hook after you are done! -- good habit!
    model.reset_hooks()

    return ActivationCache(cache, model)


def generate_and_cache(model, tokens, max_new_tokens=100, do_sample=False):
    # Always reset hook before you start -- good habit!
    model.reset_hooks()

    # (1) Create filter for what to cache
    activation_filter = lambda name: "resid_post" in name

    # (2) Construct hook functions

    # forward cache hook: cache the activation during forward run
    cache = {}

    def forward_cache_hook(act, hook, len_prompt):
        # only cache generation result
        # print(act.shape)
        if act.shape[1] == len_prompt:
            cache[hook.name] = act[:, -1, :].detach()

    # (3) Construct hooks
    temp_hook = partial(forward_cache_hook, len_prompt=tokens.shape[1])

    # (4) Run forward pass
    model.add_hook(activation_filter, temp_hook, "fwd")

    if do_sample:
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.1,
            freq_penalty=1.0,
            stop_at_eos=False,
            prepend_bos=model.cfg.default_prepend_bos,
        )
    else:
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Always reset hook after you are done! -- good habit!
    model.reset_hooks()

    return output, ActivationCache(cache, model)
