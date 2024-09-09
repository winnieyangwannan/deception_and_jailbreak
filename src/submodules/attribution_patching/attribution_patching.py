import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
import os
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

import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern


class AttributionPatching:
    def __init__(self, cfg, model, dataset):
        self.cfg = cfg
        self.model = model
        self.dataset = dataset

        (
            self.prompt_tokens_positive,
            self.answer_tokens_positive_correct,
            self.answer_tokens_positive_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            dataset.prompts_positive,
            dataset.answers_positive_correct,
            dataset.answers_positive_wrong,
        )
        (
            self.prompt_tokens_negative,
            self.answer_tokens_negative_correct,
            self.answer_tokens_negative_wrong,
        ) = tokenize_prompts_and_answers(
            model,
            dataset.prompts_negative,
            dataset.answers_negative_correct,
            dataset.answers_negative_wrong,
        )
        self.positive_baseline = None
        self.negative_baseline = None

        self.head_names = [
            f"L{l}H{h}"
            for l in range(self.model.cfg.n_layers)
            for h in range(self.model.cfg.n_heads)
        ]
        self.head_names_signed = [
            f"{name}{sign}" for name in self.head_names for sign in ["+", "-"]
        ]
        self.head_nams_qkv = [
            f"{name}{act_name}"
            for name in self.head_names
            for act_name in ["Q", "K", "V"]
        ]
        print(self.head_names[:5])
        print(self.head_names_signed[:5])
        print(self.head_nams_qkv[:5])

        self.positive_cache = None
        self.negative_cache = None
        self.positive_grad_cache = None
        self.negative_grad_cache = None

    def get_baseline_logit_diff(self) -> None:
        """
        get baseline logit differnece for lean and negative inputs
        :return:
        """
        positive_logits, negative_logits, positive_cache, negative_cache = (
            run_with_cache_contrastive(
                self.model, self.prompt_tokens_positive, self.prompt_tokens_negative
            )
        )
        positive_logit_diff = logits_to_logit_diff(
            positive_logits,
            self.answer_tokens_positive_correct,
            self.answer_tokens_positive_wrong,
        ).item()
        print(f"positive logit diff: {positive_logit_diff:.4f}")

        negative_logit_diff = logits_to_logit_diff(
            negative_logits,
            self.answer_tokens_positive_correct,
            self.answer_tokens_positive_wrong,
        ).item()
        print(f"negative logit diff: {negative_logit_diff:.4f}")

        self.positive_baseline = positive_logit_diff
        self.negative_baseline = negative_logit_diff

        print(
            f"positive Baseline is 1: {self.contrastive_metric(positive_logits).item():.4f}"
        )
        print(
            f"negative Baseline is 0: {self.contrastive_metric(negative_logits).item():.4f}"
        )

    def contrastive_metric(
        self,
        logits: Float[Tensor, "batch seq d_vocab"],
    ) -> Tensor:
        """
        Scale the logit difference between 0 and 1 (clean baseline will have logit diff =1, and negative baseline will have logit diff =0)
        :param logits:
        :return:
        """

        return (
            logits_to_logit_diff(
                logits,
                self.answer_tokens_positive_correct,
                self.answer_tokens_positive_wrong,
            )
            - self.negative_baseline
        ) / (self.positive_baseline - self.negative_baseline)

    def get_contrastive_activation_and_gradient_cache(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """
        positive_value, positive_cache, positive_grad_cache = get_cache_fwd_and_bwd(
            self.model, self.contrastive_metric, self.prompt_tokens_positive
        )
        negative_value, negative_cache, negative_grad_cache = get_cache_fwd_and_bwd(
            self.model, self.contrastive_metric, self.prompt_tokens_negative
        )
        self.positive_baseline = positive_value
        self.negative_baseline = negative_value
        self.positive_cache = positive_cache
        self.negative_cache = negative_cache
        self.positive_grad_cache = positive_grad_cache
        self.negative_grad_cache = negative_grad_cache

    def get_attention_attribution(
        self,
    ) -> Float[Tensor, "batch layer head_inde dest src"]:

        attention_stack = torch.stack(
            [self.positive_cache["pattern", l] for l in range(self.model.cfg.n_layers)],
            dim=0,
        )
        attention_grad_stack = torch.stack(
            [
                self.positive_grad_cache["pattern", l]
                for l in range(self.model.cfg.n_layers)
            ],
            dim=0,
        )
        attention_attr = attention_grad_stack * attention_stack
        attention_attr = einops.rearrange(
            attention_attr,
            "layer batch head_index dest src -> batch layer head_index dest src",
        )

        return attention_attr

    def plot_attention_attr(
        self, attention_attr, tokens, top_k=20, index=0, title="", save_path=None
    ):
        # save the visualization as .html file
        vis_path = save_path + os.sep + "attention_attribution"
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

        if len(tokens.shape) == 2:
            tokens = tokens[index]
        if len(attention_attr.shape) == 5:
            attention_attr = attention_attr[index]
        attention_attr_pos = attention_attr.clamp(min=-1e-5)
        attention_attr_neg = -attention_attr.clamp(max=1e-5)
        attention_attr_signed = torch.stack(
            [attention_attr_pos, attention_attr_neg], dim=0
        )
        attention_attr_signed = einops.rearrange(
            attention_attr_signed,
            "sign layer head_index dest src -> (layer head_index sign) dest src",
        )
        attention_attr_signed = attention_attr_signed / attention_attr_signed.max()
        attention_attr_indices = (
            attention_attr_signed.max(-1).values.max(-1).values.argsort(descending=True)
        )
        # print(attention_attr_indices.shape)
        # print(attention_attr_indices)
        attention_attr_signed = attention_attr_signed[attention_attr_indices, :, :]
        head_labels = [self.head_names_signed[i.item()] for i in attention_attr_indices]

        if title:
            display(Markdown("### " + title))
        str_tokens = self.model.to_str_tokens(tokens)
        attention = attention_attr_signed.permute(1, 2, 0)[:, :, :top_k]
        head_labels = head_labels[:top_k]

        results = {
            "str_tokens": str_tokens,
            "attention": attention,
            "head_labels": head_labels,
        }
        with open(vis_path + os.sep + "attention_attribution.pkl", "wb") as f:
            pickle.dump(results, f)
        # vis = pysvelte.AttentionMulti(
        #     tokens=self.model.to_str_tokens(tokens),
        #     attention=attention_attr_signed.permute(1, 2, 0)[:, :, :top_k],
        #     head_labels=head_labels[:top_k],
        # ).publish(dev=True)
        # vis.publish(vis_path + os.sep + f"top20.html", dev_host="try")
        # Display results

        vis = cv.attention.attention_patterns(
            attention=attention_attr_signed[:top_k, :, :],
            tokens=self.model.to_str_tokens(tokens[:]),
            attention_head_names=head_labels[:top_k],
        )

        # save the visualization as .html file
        with open(vis_path + os.sep + f"attention_attribution.html", "w") as f:
            f.write(vis._repr_html_())

    def attr_patch_residual(
        self,
    ) -> TT["component", "pos"]:
        positive_residual, residual_labels = self.positive_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=True
        )
        negative_residual = self.negative_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=False
        )
        negative_grad_residual = self.negative_grad_cache.accumulated_resid(
            -1, incl_mid=True, return_labels=False
        )
        # positive_residual shape [component batch seq d_model]
        residual_attr = einops.reduce(
            negative_grad_residual * (positive_residual - negative_residual),
            "component batch pos d_model -> component pos",
            "sum",
        )

        # Plot
        imshow(
            residual_attr,
            y=residual_labels,
            yaxis="Component",
            xaxis="Position",
            title="Residual Attribution Patching",
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_positive[0, :])
        print(f"example string token: {string_0}")
        return residual_attr, residual_labels


def run_with_cache_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits, positive_cache = model.run_with_cache(positive_tokens)
    negative_logits, negative_cache = model.run_with_cache(negative_tokens)

    return positive_logits, negative_logits, positive_cache, negative_cache


def get_cache_fwd_and_bwd(model, metric, tokens):
    # Always reset hook before you start -- good habit!
    model.reset_hooks()

    # (1) Create filter for what to cache
    filter_not_qkv_input = lambda name: "_input" not in name

    # (2) Construct hook functions

    # forward cache hook: cache the activation during forward run
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    # backward cache hook: : cache the activation during backward run
    grad_cache = {}

    def backward_cache_hook(act, hook):
        grad_cache[hook.name] = act.detach()

    # (3) Construct hooks
    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")

    # (4) Run forward pass
    logits = model(tokens)

    # (5) Get loss function (our contrastive metric)
    value = metric(logits)

    # (6) Backward pass
    value.backward()  # the loss for computing the backward pass

    # Always reset hook after you are done! -- good habit!
    model.reset_hooks()

    return (
        value.item(),
        ActivationCache(cache, model),
        ActivationCache(grad_cache, model),
    )
