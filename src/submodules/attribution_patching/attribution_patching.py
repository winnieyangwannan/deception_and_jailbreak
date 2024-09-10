import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
from fancy_einsum import einsum

import os
from typing_extensions import Literal

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

        print("test positive prompt")
        utils.test_prompt(
            self.dataset.prompts_positive[0],
            self.dataset.answers_positive_correct[0],
            self.model,
            prepend_bos=True,
        )

        print("test negative prompt")
        utils.test_prompt(
            self.dataset.prompts_negative[0],
            self.dataset.answers_negative_correct[0],
            self.model,
            prepend_bos=True,
        )

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
        Scale the logit difference between 0 and 1 (positive baseline will have logit diff =1, and negative baseline will have logit diff =0)
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

    def attr_patch_layer_out(
        self,
    ) -> TT["component", "pos"]:
        positive_layer_out, labels = self.positive_cache.decompose_resid(
            -1, return_labels=True
        )
        negative_layer_out = self.negative_cache.decompose_resid(
            -1, return_labels=False
        )
        negative_grad_layer_out = self.negative_grad_cache.decompose_resid(
            -1, return_labels=False
        )
        layer_out_attr = einops.reduce(
            negative_grad_layer_out * (positive_layer_out - negative_layer_out),
            "component batch pos d_model -> component pos",
            "sum",
        )

        imshow(
            layer_out_attr,
            y=labels,
            yaxis="Component",
            xaxis="Position",
            title="Layer Output Attribution Patching",
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_positive[0, :])
        print(f"example string token: {string_0}")

        return layer_out_attr, labels

    def attr_patch_head_out(
        self,
    ) -> TT["component", "pos"]:
        positive_layer_out, labels = self.positive_cache.stack_head_results(
            -1, return_labels=True
        )
        negative_layer_out = self.negative_cache.stack_head_results(
            -1, return_labels=False
        )
        negative_grad_layer_out = self.negative_grad_cache.stack_head_results(
            -1, return_labels=False
        )
        head_out_attr = einops.reduce(
            negative_grad_layer_out * (positive_layer_out - negative_layer_out),
            "component batch pos d_model -> component pos",
            "sum",
        )

        imshow(
            head_out_attr,
            y=labels,
            yaxis="Component",
            xaxis="Position",
            title="Head Output Attribution Patching",
        )

        sum_head_out_attr = einops.reduce(
            head_out_attr,
            "(layer head) pos -> layer head",
            "sum",
            layer=self.model.cfg.n_layers,
            head=self.model.cfg.n_heads,
        )
        imshow(
            sum_head_out_attr,
            yaxis="Layer",
            xaxis="Head Index",
            title="Head Output Attribution Patching Sum Over Pos",
        )
        string_0 = self.model.to_str_tokens(self.prompt_tokens_positive[0, :])
        print(f"example string token: {string_0}")

        return head_out_attr

    def attr_patch_head_kqv(self):
        head_vector_attr_dict = {}
        for activation_name, activation_name_full in [
            ("k", "Key"),
            ("q", "Query"),
            ("v", "Value"),
            ("z", "Mixed Value"),
        ]:
            display(
                Markdown(
                    f"#### {activation_name_full} Head Vector Attribution Patching"
                )
            )
            head_vector_attr_dict[activation_name], head_vector_labels = (
                self.attr_patch_head_vector(activation_name)
            )
            imshow(
                head_vector_attr_dict[activation_name],
                y=head_vector_labels,
                yaxis="Component",
                xaxis="Position",
                title=f"{activation_name_full} Attribution Patching",
            )
            sum_head_vector_attr = einops.reduce(
                head_vector_attr_dict[activation_name],
                "(layer head) pos -> layer head",
                "sum",
                layer=self.model.cfg.n_layers,
                head=self.model.cfg.n_heads,
            )
            imshow(
                sum_head_vector_attr,
                yaxis="Layer",
                xaxis="Head Index",
                title=f"{activation_name_full} Attribution Patching Sum Over Pos",
            )

    def attr_patch_head_vector(
        self,
        activation_name: Literal["q", "k", "v", "z"],
    ) -> TT["component", "pos"]:
        labels = self.head_names

        positive_head_vector = self.stack_head_vector_from_cache(
            self.positive_cache, activation_name
        )
        negative_head_vector = self.stack_head_vector_from_cache(
            self.negative_cache, activation_name
        )
        negative_grad_head_vector = self.stack_head_vector_from_cache(
            self.negative_grad_cache, activation_name
        )
        head_vector_attr = einops.reduce(
            negative_grad_head_vector * (positive_head_vector - negative_head_vector),
            "component batch pos d_head -> component pos",
            "sum",
        )
        return head_vector_attr

    def stack_head_vector_from_cache(
        self, cache, activation_name: Literal["q", "k", "v", "z"]
    ) -> TT["layer_and_head_index", "batch", "pos", "d_head"]:
        """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
        stacked_head_vectors = torch.stack(
            [cache[activation_name, l] for l in range(self.model.cfg.n_layers)], dim=0
        )
        stacked_head_vectors = einops.rearrange(
            stacked_head_vectors,
            "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
        )
        return stacked_head_vectors

    def attr_patch_head_path(
        self,
    ) -> TT["qkv", "dest_component", "src_component", "pos"]:
        """
        Computes the attribution patch along the path between each pair of heads.

        Sets this to zero for the path from any late head to any early head

        """
        start_labels = self.head_names
        end_labels = self.head_nams_qkv

        # get teh negative gradient cache per head for kqv
        full_vector_grad_late_node = self.get_full_vector_grad_late_node(
            self.negative_grad_cache
        )

        # get the positive activation cache per head
        positive_head_stack_early_node = self.positive_cache.stack_head_results(-1)
        # get the negative activation cache per head
        negative_head_stack_early_node = self.negative_cache.stack_head_results(-1)

        # difference in positive and negative
        diff_head_early_node = einops.rearrange(
            positive_head_stack_early_node - negative_head_stack_early_node,
            "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
            layer=self.model.cfg.n_layers,
            head_index=self.model.cfg.n_heads,
        )
        # multiply the difference by the gradeint
        path_attr = einsum(
            "qkv layer_end batch pos head_end d_model, layer_start batch pos head_start d_model -> qkv layer_end head_end layer_start head_start pos",
            full_vector_grad_late_node,
            diff_head_early_node,
        )

        # what does this do???
        correct_layer_order_mask = (
            torch.arange(self.model.cfg.n_layers)[None, :, None, None, None, None]
            > torch.arange(self.model.cfg.n_layers)[None, None, None, :, None, None]
        ).to(path_attr.device)

        # replace late to early with zero
        zero = torch.zeros(1, device=path_attr.device)
        path_attr = torch.where(
            correct_layer_order_mask, path_attr, zero
        )  # condition, input, other

        # rearrange to 3d
        path_attr = einops.rearrange(
            path_attr,
            "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
        )
        imshow(
            path_attr.sum(-1),  # sum over all position
            y=end_labels,
            yaxis="Path End (Head Input)",
            x=start_labels,
            xaxis="Path Start (Head Output)",
            title="Head Path Attribution Patching",
        )
        return path_attr

    def get_head_vector_grad_input_from_grad_cache(
        self,
        grad_cache: ActivationCache,
        activation_name: Literal["q", "k", "v"],
        layer: int,
    ) -> Float[Tensor, "batch pos head_index d_model"]:
        """
        We want to get the gradient with respect to key multiply by key.
        What we got is the residual stream

        key = residual_stream @ W_K + b_K, then residual_stream_grad_mediated_by_key = W_K @ key
        :param grad_cache:
        :param activation_name:
        :param layer:
        :return:
        """
        vector_grad = grad_cache[activation_name, layer]  # get access to the k,q, v
        ln_scales = grad_cache["scale", layer, "ln1"]  # get access to scale
        attn_layer_object = self.model.blocks[layer].attn
        if activation_name == "q":
            W = attn_layer_object.W_Q
        elif activation_name == "k":
            W = attn_layer_object.W_K
        elif activation_name == "v":
            W = attn_layer_object.W_V
        else:
            raise ValueError("Invalid activation name")

        return einsum(
            "batch pos head_index d_head, batch pos head_index, head_index d_model d_head -> batch pos head_index d_model",
            vector_grad,
            ln_scales.squeeze(-1),
            W,
        )

    def get_stacked_head_vector_grad_late_node(
        self, grad_cache, activation_name: Literal["q", "k", "v"]
    ) -> Float[Tensor, "layer batch pos head_index d_model"]:
        return torch.stack(
            [
                self.get_head_vector_grad_input_from_grad_cache(
                    grad_cache, activation_name, l
                )
                for l in range(self.model.cfg.n_layers)
            ],
            dim=0,
        )

    def get_full_vector_grad_late_node(
        self,
        grad_cache,
    ) -> Float[Tensor, "qkv layer batch pos head_index d_model"]:
        return torch.stack(
            [
                self.get_stacked_head_vector_grad_late_node(grad_cache, activation_name)
                for activation_name in ["q", "k", "v"]
            ],
            dim=0,
        )

    def top_heads_plot_path_attr(self, head_out_attr, head_path_attr):
        start_labels = self.head_names
        end_labels = self.head_nams_qkv

        head_out_values, head_out_indices = (
            head_out_attr.sum(-1).abs().sort(descending=True)
        )
        line(head_out_values)

        top_head_indices = head_out_indices[:22].sort().values
        top_end_indices = []
        top_end_labels = []
        top_start_indices = []
        top_start_labels = []
        for i in top_head_indices:
            i = i.item()
            top_start_indices.append(i)
            top_start_labels.append(start_labels[i])
            for j in range(3):
                top_end_indices.append(3 * i + j)
                top_end_labels.append(end_labels[3 * i + j])

        imshow(
            head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1),
            y=top_end_labels,
            yaxis="Path End (Head Input)",
            x=top_start_labels,
            xaxis="Path Start (Head Output)",
            title="Head Path Attribution Patching (Filtered for Top Heads)",
        )

        for j, composition_type in enumerate(["Query", "Key", "Value"]):
            imshow(
                head_path_attr[top_end_indices, :][:, top_start_indices][j::3].sum(-1),
                y=top_end_labels[j::3],
                yaxis="Path End (Head Input)",
                x=top_start_labels,
                xaxis="Path Start (Head Output)",
                title=f"Head Path to {composition_type} Attribution Patching (Filtered for Top Heads)",
            )

        # subplots
        # todo: understand this!
        top_head_path_attr = einops.rearrange(
            head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1),
            "(head_end qkv) head_start -> qkv head_end head_start",
            qkv=3,
        )
        imshow(
            top_head_path_attr,
            y=[i[:-1] for i in top_end_labels[::3]],
            yaxis="Path End (Head Input)",
            x=top_start_labels,
            xaxis="Path Start (Head Output)",
            title=f"Head Path Attribution Patching (Filtered for Top Heads)",
            facet_col=0,
            facet_labels=["Query", "Key", "Value"],
        )


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
