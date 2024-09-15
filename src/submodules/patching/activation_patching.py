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


class AttributionPatching:
    def __init__(self, contrastive_dataset, contrastive_base):

        self.model = contrastive_base.model
        self.cfg = contrastive_base.cfg

        if contrastive_base.cfg.clean_type == "positive":
            self.corrupted_tokens = contrastive_base.prompt_tokens_negative
            self.clean_tokens = contrastive_base.prompt_tokens_negative
            self.clean_cache = contrastive_base.cache_positive
            self.answer_tokens_correct = contrastive_base.answer_tokens_positive_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_positive_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean
        else:
            self.corrupted_tokens = contrastive_dataset.prompt_tokens_positive
            self.clean_tokens = contrastive_base.prompt_tokens_negative
            self.clean_cache = contrastive_base.cache_negative
            self.answer_tokens_correct = contrastive_base.answer_tokens_negative_correct
            self.answer_tokens_wrong = contrastive_base.answer_tokens_negative_wrong
            self.logit_diff_corrupted = contrastive_base.logit_diff_corrupted
            self.logit_diff_clean = contrastive_base.logit_diff_clean

    def get_activation_patching_residual(self):

        patching_metric = partial(
            logit_diff_metric,
            answer_tokens_correct=self.answer_tokens_correct,
            answer_tokens_wrong=self.answer_tokens_wrong,
            logit_diff_corrupted=self.logit_diff_corrupted,
            logit_diff_clean=self.logit_diff_clean,
        )

        act_patch_resid_pre = patching.get_act_patch_resid_pre(
            model=self.model,
            corrupted_tokens=self.corrupted_tokens,
            clean_cache=self.clean_cache,
            patching_metric=patching_metric,
        )

        labels = [
            f"{tok} {i}"
            for i, tok in enumerate(self.model.to_str_tokens(self.clean_tokens[0]))
        ]
        fig = imshow(
            act_patch_resid_pre,
            labels={"x": "Position", "y": "Layer"},
            x=labels,
            title="resid_pre Activation Patching",
            width=600,
            return_fig=True,
        )

        fig_path = os.path.join(
            self.cfg.artifact_path(), "activation_patching", "resid_pre"
        )
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        pio.write_image(
            fig,
            fig_path
            + os.sep
            + f"{self.cfg.model_alias}_resid_pre_activation_patching.png",
            scale=6,
        )
        fig.write_html(
            fig_path
            + os.sep
            + f"{self.cfg.model_alias}_resid_pre_activation_patching.html"
        )


def logit_diff_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
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
