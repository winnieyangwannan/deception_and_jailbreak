import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from anyio import sleep_forever
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
from fancy_einsum import einsum
import gc
import os
from typing_extensions import Literal
from tqdm import tqdm
from functools import partial
from src.utils.hook_utils import add_hooks
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
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import line, scatter, bar
from src.utils.plotly_utils_neel import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern
from src.submodules.evaluations.evaluate_deception import (
    evaluate_generation_deception,
    plot_lying_honest_performance,
)
from src.submodules.stage_statistics.stage_statistics import get_state_quantification
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from src.utils.hook_utils import cache_activation_hook_post, cache_residual_hook_pre
from src.utils.hook_utils import activation_addition_cache_last_post


class ContrastiveActSteering:
    def __init__(self, cfg, contrastive_dataset, contrastive_base):

        self.base = contrastive_base
        self.dataset = contrastive_dataset
        self.cfg = cfg
        self.steering_vector = None
        self.steer_cache_positive = None
        self.steer_cache_negative = None

    def get_steering_vector(self):
        mean_act_positive = self.base.cache_positive[self.cfg.source_layer, :, -1].mean(
            dim=0
        )
        mean_act_negative = self.base.cache_negative[self.cfg.source_layer, :, -1].mean(
            dim=0
        )

        if self.cfg.intervention == "positive_addition":

            self.steering_vector = mean_act_positive - mean_act_negative

    def generate_and_contrastive_activation_steering(self):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """

        steer_cache_positive = self.generate_and_steer_batch(
            self.dataset.prompts_positive,
            self.base.prompt_tokens_positive,
            self.base.prompt_masks_positive,
            self.cfg.contrastive_type[0],
        )

        steer_cache_negative = self.generate_and_steer_batch(
            self.dataset.prompts_negative,
            self.base.prompt_tokens_negative,
            self.base.prompt_masks_negative,
            self.cfg.contrastive_type[1],
        )

        self.steer_cache_positive = steer_cache_positive
        self.steer_cache_negative = steer_cache_negative
        print(
            "free(Gb):",
            torch.cuda.mem_get_info()[0] / 1000000000,
            "total(Gb):",
            torch.cuda.mem_get_info()[1] / 1000000000,
        )
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    def generate_and_steer_batch(
        self, prompts, prompt_tokens, prompt_masks, contrastive_type
    ):

        # 1. Gen data
        statements = self.dataset.statements
        labels = self.dataset.answers_positive_correct

        # 2. Generation initialization
        completions = []
        output, cache_all = self.initialize_cache(
            prompt_tokens,
            prompt_masks,
            mode="generate",
        )
        # Construct generation result for saving (one batch)
        generation_toks = output[:, prompt_tokens.shape[1] :]
        completions.append(
            {
                "prompt": prompts[0],
                "statement": statements[0],
                "response": self.base.model.tokenizer.decode(
                    generation_toks[0], skip_special_tokens=True
                ).strip(),
                "label": labels[0],
                "ID": 0,
            }
        )
        # 3. Main loop
        n_prompts = len(prompt_tokens)
        for batch in tqdm(range(1, n_prompts, self.cfg.batch_size)):

            output, cache = generate_and_steer(
                self.cfg,
                self.base.model,
                self.steering_vector,
                prompt_tokens[batch : batch + self.cfg.batch_size],
                masks=prompt_masks[batch : batch + self.cfg.batch_size],
            )

            cache_all = torch.cat((cache_all, cache), dim=1)

            # Construct generation result for saving (one batch)
            generation_toks = output[:, prompt_tokens.shape[1] :]
            for generation_idx, generation in enumerate(generation_toks):
                completions.append(
                    {
                        "statement": statements[batch + generation_idx],
                        "response": self.base.model.tokenizer.decode(
                            generation, skip_special_tokens=True
                        ).strip(),
                        "label": labels[batch + generation_idx],
                        "ID": batch + generation_idx,
                    }
                )

        # 3. Save completion
        completion_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "completions",
        )
        if not os.path.exists(completion_path):
            os.makedirs(completion_path)
        with open(
            completion_path
            + os.sep
            + f"{self.cfg.model_alias}_completions_source_{self.cfg.source_layer}_target_{self.cfg.target_layer}_{contrastive_type}.json",
            "w",
        ) as f:
            json.dump(completions, f, indent=4)

        return cache_all

    def initialize_cache(self, prompt_tokens, prompt_masks, mode="run"):

        output, cache = generate_and_steer(
            self.cfg,
            self.base.model,
            self.steering_vector,
            prompt_tokens[0 : 0 + 1],
            masks=prompt_masks[0 : 0 + 1],
        )

        return output, cache

    def get_contrastive_activation_pca(self, pos=-1):

        # 1. Get access to cache
        activations_positive, activations_negative = self.get_contrastive_activations()

        # 2. Get pca and plot
        fig = get_pca_and_plot(
            self.cfg, activations_positive, activations_negative, self.dataset
        )

        # 3. Save plot
        pca_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "pca",
        )
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(
            pca_path
            + os.sep
            + f"{self.cfg.model_alias}_pca_source_{self.cfg.source_layer}_target_{self.cfg.target_layer}.html"
        )
        pio.write_image(
            fig,
            pca_path
            + os.sep
            + f"{self.cfg.model_alias}_pca_source_{self.cfg.source_layer}_target_{self.cfg.target_layer}.png",
            scale=6,
        )

        # 4. Save activations
        activations = {
            "activations_positive": activations_positive,
            "activations_negative": activations_negative,
            "labels": self.dataset.labels_positive_correct,
        }
        activation_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "activations",
        )
        if not os.path.exists(activation_path):
            os.makedirs(activation_path)
        with open(
            activation_path
            + os.sep
            + f"{self.cfg.model_alias}_activations_source_{self.cfg.source_layer}_target_{self.cfg.target_layer}.pkl",
            "wb",
        ) as f:
            pickle.dump(activations, f)

    def evaluate_deception(self):
        completion_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "completions",
        )
        save_name = f"source_{self.cfg.source_layer}_target_{self.cfg.target_layer}"
        completions_honest = evaluate_generation_deception(
            self.cfg,
            contrastive_type=self.cfg.contrastive_type[0],
            save_path=completion_path,
            save_name=save_name,
        )
        completions_lying = evaluate_generation_deception(
            self.cfg,
            contrastive_type=self.cfg.contrastive_type[1],
            save_path=completion_path,
            save_name=save_name,
        )
        performance_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "performance",
        )
        plot_lying_honest_performance(
            self.cfg,
            completions_honest,
            completions_lying,
            save_path=performance_path,
            save_name=save_name,
        )

    def get_stage_statistics(self):
        save_name = f"source_{self.cfg.source_layer}_target_{self.cfg.target_layer}"
        # 1. Get access to cache
        activations_positive, activations_negative = self.get_contrastive_activations()

        labels = self.dataset.labels_positive_correct

        # 2. get stage statistics
        get_state_quantification(
            self.cfg,
            activations_positive,
            activations_negative,
            labels,
            save_plot=True,
            save_name=save_name,
        )

    def get_contrastive_activations(self):
        if self.cfg.framework == "transformer_lens":
            activations_positive = [
                self.steer_cache_positive[
                    utils.get_act_name(self.cfg.cache_name, l)
                ].squeeze()
                for l in range(self.base.model.cfg.n_layers)
            ]

            activations_negative = [
                self.steer_cache_negative[
                    utils.get_act_name(self.cfg.cache_name, l)
                ].squeeze()
                for l in range(self.base.model.cfg.n_layers)
            ]
            # stack the list of activations # shape = [batch,d_model]
            activations_positive = torch.stack(activations_positive)
            activations_negative = torch.stack(activations_negative)
        else:
            activations_positive = self.steer_cache_positive
            activations_negative = self.steer_cache_positive
        return activations_positive, activations_negative

    # def get_logit_diff(self, logits_positive, logits_negative):
    #     if self.cfg.clean_type == "positive":
    #         answer_tokens_correct = self.answer_tokens_positive_correct
    #         answer_tokens_wrong = self.answer_tokens_positive_wrong
    #     else:
    #         answer_tokens_correct = self.answer_tokens_negative_correct
    #         answer_tokens_wrong = self.answer_tokens_negative_wrong
    #
    #     logits_diff_positive = logits_to_ave_logit_diff(
    #         logits_positive,
    #         answer_tokens_correct,
    #         answer_tokens_wrong,
    #     )
    #     logits_diff_negative = logits_to_ave_logit_diff(
    #         logits_negative,
    #         answer_tokens_correct,
    #         answer_tokens_wrong,
    #     )

    # if self.cfg.clean_type == "positive":
    #
    #     self.logit_diff_clean = logits_diff_positive
    #     self.logit_diff_corrupted = logits_diff_negative
    # else:
    #     self.logtis_diff_clean = logits_diff_negative
    #     self.logit_diff_corrupted = logits_diff_positive
    #
    # print(f"logit diff clean: {self.logit_diff_clean}")
    # print(f"logit diff corrupted: {self.logit_diff_corrupted}")
    # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="/n/n")


def run_with_cache_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits, cache_positive = model.run_with_cache(positive_tokens)
    negative_logits, cache_negative = model.run_with_cache(negative_tokens)

    return positive_logits, negative_logits, cache_positive, cache_negative


def run_model_contrastive(model, positive_tokens, negative_tokens):

    # Run the model and cache all activations
    positive_logits = model(positive_tokens)
    negative_logits = model(negative_tokens)

    return positive_logits, negative_logits


def generate_and_steer(cfg, model, steering_vector, tokens, masks=None):

    max_new_tokens = cfg.max_new_tokens
    do_sample = cfg.do_sample
    n_layers = model.model.config.num_hidden_layers
    d_model = model.model.config.hidden_size
    n_samples = len(tokens)
    block_modules = model.model_block_modules
    len_prompt = tokens.shape[1]
    cache = torch.zeros(
        (n_layers, n_samples, d_model),
        dtype=torch.float16,
        device=model.device,
    )

    fwd_hook = [
        (
            block_modules[layer],
            activation_addition_cache_last_post(
                mean_diff=steering_vector,
                cache=cache,
                target_layer=cfg.target_layer,
                layer=layer,
                len_prompt=len_prompt,
                pos=-1,  # only ca
            ),
        )
        for layer in range(n_layers)
    ]

    # (4) Run forward pass
    if do_sample:

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.1,
            repetition_penalty=1.0,
        )
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hook):

            output = model.model.generate(
                input_ids=tokens,
                attention_mask=masks,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )

    else:

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hook):

            output = model.model.generate(
                input_ids=tokens,
                attention_mask=masks,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )

    # (5) Output

    return output, cache
