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
)
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import line, scatter, bar
from src.utils.plotly_utils_neel import imshow
from src.submodules.activation_pca.activation_pca import get_pca_and_plot
import plotly.io as pio
import plotly.graph_objects as go
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
    def __init__(
        self,
        cfg,
        contrastive_dataset,
        contrastive_base,
        source_layer=None,
        target_layer=None,
    ):

        self.base = contrastive_base
        self.dataset = contrastive_dataset
        self.cfg = cfg
        self.steering_vector = None
        self.steer_cache_positive = None
        self.steer_cache_negative = None
        self.source_layer = source_layer
        self.target_layer = target_layer
        if self.source_layer is None:
            self.source_layer = self.cfg.source_layer
            self.target_layer = self.cfg.target_layer
        print(f"self.source_layer{self.source_layer}")
        print(f"self.target_layer{self.target_layer}")

    def get_steering_vector(self):

        if self.cfg.framework == "transformer_lens":
            mean_act_positive = self.base.cache_positive[
                utils.get_act_name(self.cfg.cache_name, self.source_layer)
            ][:, -1].mean(dim=0)
            mean_act_negative = self.base.cache_negative[
                utils.get_act_name(self.cfg.cache_name, self.source_layer)
            ][:, -1].mean(dim=0)
        else:
            mean_act_positive = self.base.cache_positive[self.source_layer, :, -1].mean(
                dim=0
            )
            mean_act_negative = self.base.cache_negative[self.source_layer, :, -1].mean(
                dim=0
            )

        if self.cfg.intervention == "positive_addition":

            self.steering_vector = mean_act_positive - mean_act_negative
        return self.steering_vector

    def generate_and_contrastive_activation_steering(self, steering_vector):
        """
        Get the activation cache and gradient cache for both the positive and negative prompts
        :return:
        """
        self.steering_vector = steering_vector
        if self.cfg.framework == "transformer_lens":
            steer_cache_positive = self.generate_and_steer_batch_tl(
                self.dataset.prompts_positive,
                self.base.prompt_tokens_positive,
                self.cfg.contrastive_type[0],
            )

            steer_cache_negative = self.generate_and_steer_batch_tl(
                self.dataset.prompts_negative,
                self.base.prompt_tokens_negative,
                self.cfg.contrastive_type[1],
            )
        else:

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

            output, cache = self.generate_and_steer(
                self.base.model,
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
            + f"{self.cfg.model_alias}_completions_source_{self.source_layer}_target_{self.target_layer}_{contrastive_type}.json",
            "w",
        ) as f:
            json.dump(completions, f, indent=4)

        return cache_all

    def generate_and_steer_batch_tl(self, prompts, prompt_tokens, contrastive_type):

        # 1. Gen data
        statements = self.dataset.statements
        labels = self.dataset.answers_positive_correct

        # 2. Generation initialization
        completions = []
        output, cache_all = self.initialize_cache(
            prompt_tokens,
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

            output, cache = self.generate_and_steer_tl(
                self.base.model,
                prompt_tokens[batch : batch + self.cfg.batch_size],
            )
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
            + f"{self.cfg.model_alias}_completions_source_{self.source_layer}_target_{self.target_layer}_{contrastive_type}.json",
            "w",
        ) as f:
            json.dump(completions, f, indent=4)

        return cache_all

    def generate_and_steer(self, model, tokens, masks=None):

        max_new_tokens = self.cfg.max_new_tokens
        do_sample = self.cfg.do_sample
        n_layers = model.model.config.num_hidden_layers
        d_model = model.model.config.hidden_size
        n_samples = len(tokens)
        block_modules = model.model_block_modules
        len_prompt = tokens.shape[1]
        cache = torch.zeros(
            (n_layers, n_samples, d_model),
            dtype=torch.float64,
            device=model.device,
        )

        fwd_hook = [
            (
                block_modules[layer],
                activation_addition_cache_last_post(
                    mean_diff=self.steering_vector,
                    cache=cache,
                    target_layer=self.target_layer,
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

    def generate_and_steer_tl(self, model, tokens, masks=None):
        model.reset_hooks()
        cache = {}

        max_new_tokens = self.cfg.max_new_tokens
        n_layers = model.cfg.n_layers
        do_sample = self.cfg.do_sample
        len_prompt = tokens.shape[1]

        def steering(
            activations,
            hook,
            steering_vector=None,
            patch_layer=None,
            len_prompt=None,
            cfg=None,
            steering_strength=1.0,
        ):
            # print(steering_vector.shape) # [batch, n_tokens, n_head, d_head ]
            # print("hook.layer()")
            # print(hook.layer())
            if cfg.cache_pos == "prompt_last_token":
                if activations.shape[1] == len_prompt:
                    cache[hook.name] = activations[:, -1, :].detach()

            if hook.layer() == patch_layer:
                if cfg.intervention == "positive_addition":
                    return activations + steering_strength * steering_vector

        steering_hook = partial(
            steering,
            steering_vector=self.steering_vector,
            patch_layer=self.target_layer,
            len_prompt=len_prompt,
            cfg=self.cfg,
            steering_strength=1,
        )

        # (4) Run forward pass
        if do_sample:
            # [utils.get_act_name(receiver_input, layer) for layer in receiver_layers]
            with model.hooks(
                fwd_hooks=[
                    (
                        utils.get_act_name(self.cfg.cache_name, layer),
                        steering_hook,
                    )
                    for layer in range(n_layers)
                    # (
                    #     utils.get_act_name(self.cfg.cache_name, self.cfg.target_layer),
                    #     steering_hook,
                    # )
                ]
            ):
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

        # (5) Output
        model.reset_hooks()

        return output, ActivationCache(cache, model)

    def initialize_cache(self, prompt_tokens, prompt_masks=None):

        if self.cfg.framework == "transformer_lens":
            output, cache = self.generate_and_steer_tl(
                self.base.model,
                prompt_tokens[0 : 0 + 1],
            )
        else:
            output, cache = self.generate_and_steer(
                self.base.model,
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
            + f"{self.cfg.model_alias}_pca_source_{self.source_layer}_target_{self.target_layer}.html"
        )
        pio.write_image(
            fig,
            pca_path
            + os.sep
            + f"{self.cfg.model_alias}_pca_source_{self.source_layer}_target_{self.target_layer}.png",
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
            + f"{self.cfg.model_alias}_activations_source_{self.source_layer}_target_{self.target_layer}.pkl",
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
        save_name = f"source_{self.source_layer}_target_{self.target_layer}"
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
        save_name = f"source_{self.source_layer}_target_{self.target_layer}"
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
            activations_negative = self.steer_cache_negative
        return activations_positive, activations_negative


class ContrastiveActSteeringAll:
    def __init__(self, cfg, model_base):
        self.cfg = cfg
        self.model = model_base

    def get_honest_score_all_layers(self):
        n_layers = self.model.cfg.n_layers
        # 1. Load performance
        acc_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "completions",
        )

        score_positive_all = []
        score_negative_all = []
        for layer in range(self.model.cfg.n_layers):
            save_name = f"source_{layer}_target_{layer}"

            with open(
                acc_path
                + os.sep
                + f"{self.cfg.model_alias}_completions_{save_name}_{self.cfg.contrastive_type[0]}.json",
                "r",
            ) as f:
                completion_honest = json.load(f)
                score_positive = completion_honest["honest_score"]
                score_positive_all.append(score_positive)
            with open(
                acc_path
                + os.sep
                + f"{self.cfg.model_alias}_completions_{save_name}_{self.cfg.contrastive_type[1]}.json",
                "r",
            ) as f:
                completion_lying = json.load(f)
                score_negative = completion_lying["honest_score"]
                score_negative_all.append(score_negative)

        # save
        score_all = {
            "score_positive_all": score_positive_all,
            "score_negative_all": score_negative_all,
        }
        performance_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "performance",
        )
        with open(
            performance_path
            + os.sep
            + f"{self.cfg.model_alias}_honest_score_all_layers.pkl",
            "wb",
        ) as f:
            pickle.dump(score_all, f)

        # plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(self.model.cfg.n_layers),
                y=score_positive_all,
                name="Lying",
                mode="lines+markers",
                marker=dict(color="dodgerblue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(self.model.cfg.n_layers),
                y=score_negative_all,
                name="Honest",
                mode="lines+markers",
                marker=dict(color="gold"),
            )
        )
        fig.update_layout(
            xaxis_title="Layer", yaxis_title="Accuracy", width=600, height=300
        )
        fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
        fig.show()
        fig.write_html(performance_path + os.sep + "honest_score_summary" + ".html")
        pio.write_image(
            fig,
            performance_path + os.sep + "honest_score_summary" + ".png",
            scale=6,
        )

        return score_positive_all, score_negative_all

    def get_cos_sim_all_layers(self):

        # 1. Load
        cos_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_steering",
            self.cfg.intervention,
            "stage_stats",
        )

        cos_all = []
        for layer in range(self.model.cfg.n_layers):
            save_name = f"source_{layer}_target_{layer}"
            with open(
                cos_path
                + os.sep
                + f"{self.cfg.model_alias}_stage_stats_{save_name}.pkl",
                "r",
            ) as f:
                data = json.load(f)
                cos = data["stage_3"]["cos_positive_negative"]
                cos_all.append(cos)

        # save
        score_all = {
            "cos_all": cos_all,
        }

        with open(
            cos_path + os.sep + f"{self.cfg.model_alias}_cos_sim_all_layers.pkl",
            "wb",
        ) as f:
            pickle.dump(score_all, f)

        return cos_all

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
