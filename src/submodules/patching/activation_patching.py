import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
from src.utils.utils import logits_to_ave_logit_diff

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
from src.submodules.model.load_model import load_model
from src.utils.utils import logits_to_ave_logit_diff
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
from src.submodules.stage_statistics.stage_statistics import get_state_quantification
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformer_lens import patching
from transformer_lens.hook_points import HookPoint
from src.utils.hook_utils import add_hooks
from src.utils.hook_utils import patch_residual_hook_pre, patch_residual_hook_post


class AttributionPatching:
    def __init__(self, contrastive_dataset, contrastive_base):

        self.model = contrastive_base.model
        self.cfg = contrastive_base.cfg

        if contrastive_base.cfg.clean_type == "positive":
            self.corrupted_tokens = contrastive_base.prompt_tokens_negative
            self.corrupted_masks = contrastive_base.prompt_masks_negative
            self.clean_tokens = contrastive_base.prompt_tokens_positive
            self.clean_masks = contrastive_base.prompt_masks_positive
            self.clean_cache = contrastive_base.cache_positive
            self.clean_prompts = contrastive_dataset.prompts_positive
            self.corrupted_prompts = contrastive_dataset.prompts_negative
            self.answer_tokens_correct = contrastive_base.answer_tokens_positive_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_positive_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean
        else:
            self.corrupted_tokens = contrastive_dataset.prompt_tokens_positive
            self.corrupted_masks = contrastive_dataset.prompt_masks_positive
            self.clean_tokens = contrastive_base.prompt_tokens_negative
            self.clean_masks = contrastive_base.prompt_masks_negative
            self.clean_cache = contrastive_base.cache_negative
            self.clean_prompts = contrastive_dataset.prompts_postive
            self.corrupted_prompts = contrastive_dataset.prompts_negative
            self.answer_tokens_correct = contrastive_base.answer_tokens_negative_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_negative_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean

    def get_activation_patching_block(self):
        """ "
        # patch component including: resid pre, resid mid, resid post, mlp_out, attn_out
        """

        save_path = os.path.join(
            self.cfg.artifact_path(),
            "activation_patching",
            "residual",
            self.cfg.framework,
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        patching_metric = partial(
            logit_diff_metric,
            answer_tokens_correct=self.answer_tokens_correct,
            answer_tokens_wrong=self.answer_tokens_wrong,
            logit_diff_corrupted=self.logit_diff_corrupted,
            logit_diff_clean=self.logit_diff_clean,
        )

        if self.cfg.framework == "transformer_lens":
            act_patch = patching.get_act_patch_resid_pre(
                model=self.model,
                corrupted_tokens=self.corrupted_tokens,
                clean_cache=self.clean_cache,
                patching_metric=patching_metric,
            )
        else:
            act_patch = self.act_patch_block(patching_metric=patching_metric)

        # Plot
        if self.cfg.framework == "transformer_lens":
            prompt = self.clean_tokens[0]
        else:
            prompt = self.model.tokenizer.decode(self.clean_tokens[0])
        labels = [
            f"{tok} {i}" for i, tok in enumerate(self.model.to_str_tokens(prompt))
        ]

        fig = imshow(
            act_patch,
            x=labels,
            title="Logit Difference From Patched Residual Stream",
            labels={"x": "Sequence Position", "y": "Layer"},
            width=600,  # If you remove this argument, the plot will usually fill the available space
            return_fig=True,
        )
        # save fig
        pio.write_image(
            fig,
            save_path
            + os.sep
            + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.png",
            scale=6,
        )
        fig.write_html(
            save_path
            + os.sep
            + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.html"
        )

        # Save results
        with open(
            save_path
            + os.sep
            + f"{self.cfg.model_alias}_{self.cfg.cache_name}_activation_patching.pkl",
            "wb",
        ) as f:
            pickle.dump(act_patch, f)

    def act_patch_block(
        self, patching_metric: Callable[[Float[Tensor, "batch d_vocab"]], float]
    ) -> Float[Tensor, "layer pos"]:
        """
        Returns an array of results of patching each position at each layer in the residual
        stream, using the value from the clean cache.

        The results are calculated using the patching_metric function, which should be
        called on the model's logit output.
        """
        n_layers = self.model.cfg.n_layers
        seq_len = self.corrupted_tokens.size(-1)
        results = torch.zeros(n_layers, seq_len, device=self.model.device)

        for patch_layer in tqdm(range(n_layers)):
            for patch_pos in range(seq_len):
                patched_logit = self.patch_residual_component(
                    patch_layer=patch_layer,
                    patch_pos=patch_pos,
                )
                results[patch_layer, patch_pos] = patching_metric(patched_logit)

        return results

    def patch_residual_component(
        self,
        patch_layer: int,
        patch_pos: int,
    ) -> Float[Tensor, "batch d_model"]:
        """
        Patches a given sequence position in the residual stream, using the value
        from the clean cache.
        """

        # (1) Get forward hook
        block_modules = self.model.model_block_modules
        n_layers = self.model.model.config.num_hidden_layers
        if self.cfg.cache_name == "resid_pre":
            fwd_hook_pre = [
                (
                    block_modules[layer],
                    patch_residual_hook_pre(
                        clean_cache=self.clean_cache,
                        patch_layer=patch_layer,
                        layer=layer,
                        patch_pos=patch_pos,
                    ),
                )
                for layer in range(n_layers)
            ]
            fwd_hook_post = []
        else:
            fwd_hook_post = [
                (
                    block_modules[layer],
                    patch_residual_hook_post(
                        clean_cache=self.clean_cache,
                        patch_layer=patch_layer,
                        layer=layer,
                        patch_pos=patch_pos,
                    ),
                )
                for layer in range(n_layers)
            ]
            fwd_hook_pre = []

        # (3) run with hook
        with add_hooks(
            module_forward_pre_hooks=fwd_hook_pre, module_forward_hooks=fwd_hook_post
        ):
            patched_logit = self.model.model(
                input_ids=self.corrupted_tokens, attention_mask=self.corrupted_masks
            )

        return patched_logit[0][:, -1, :]


def logit_diff_metric(
    logits: Float[Tensor, "batch d_vocab"],
    answer_tokens_correct: Float[Tensor, "batch"],
    answer_tokens_wrong: Float[Tensor, "batch"],
    logit_diff_corrupted: float,
    logit_diff_clean: float,
) -> Float[Tensor, ""]:
    """
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on corrupted input, and 1 when performance is same as on clean input.
    """
    patched_logit_diff = logits_to_ave_logit_diff(
        logits, answer_tokens_correct, answer_tokens_wrong
    )

    return (patched_logit_diff - logit_diff_corrupted) / (
        logit_diff_clean - logit_diff_corrupted
    )
