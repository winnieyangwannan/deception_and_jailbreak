import random
import json
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import numpy as np
from matplotlib.widgets import EllipseSelector
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
import os
import argparse
from pipeline.configs.config_generation import Config
from datasets import load_dataset
import sae_lens
from sae_lens import HookedSAETransformer
from tqdm import tqdm
from src.submodules.evaluations.run_evaluate_generation_deception import (
    plot_lying_honest_performance,
)
from src.submodules.evaluations.run_evaluate_generation_deception import (
    evaluate_generation_deception,
)

import torch
from src.submodules.dataset.task_and_dataset_deception import (
    load_and_sample_datasets,
    construct_prompt,
)
from src.submodules.model.load_model import load_model
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import imshow, line, scatter, bar
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern

torch.set_grad_enabled(False)


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")

    return parser.parse_args()


def test_prompt(cfg, model, dataset: list):
    statements = [row["claim"] for row in dataset]
    labels = [row["label"] for row in dataset]
    prompts = construct_prompt(
        cfg, statements, contrastive_label=cfg.contrastive_label[0]
    )

    example_id = 0
    example_prompt = prompts[example_id]
    if labels[example_id] == 0:
        example_answer = "false"
    else:
        example_answer = "true"

    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)


def run_with_cache(cfg, model, dataset: list[str]):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # get statement from dataset
    statements = [row["claim"] for row in dataset]
    labels = [row["label"] for row in dataset]
    prompts_positive = construct_prompt(
        cfg, statements, contrastive_label=cfg.contrastive_label[0]
    )
    prompts_negative = construct_prompt(
        cfg, statements, contrastive_label=cfg.contrastive_label[1]
    )
    prompts = prompts_positive + prompts_negative
    # From str to tokens
    tokens = model.to_tokens(prompts, prepend_bos=True)
    # Move the tokens to the GPU
    tokens = tokens.to(device)
    # Run the model and cache all activations
    logits, cache = model.run_with_cache(tokens)

    # correct answer and incorrect answer
    answers_positive = []
    for ii, label in enumerate(labels):
        if label == 1:
            answers_positive.append(("true", "false"))
        elif label == 0:
            answers_positive.append(("false", "true"))
    answers_negative = []
    for ii, label in enumerate(labels):
        if label == 1:
            answers_negative.append(("false", "true"))
        elif label == 0:
            answers_negative.append(("true", "false"))
    answers = answers_positive + answers_negative
    answer_tokens = torch.concat(
        [model.to_tokens(answer, prepend_bos=False).T for answer in answers]
    )

    return logits, cache, prompts, answers, answer_tokens


def logits_to_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False,
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    """
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    """
    final_logits = logits[:, -1, :]
    answer_logits = torch.gather(final_logits, dim=-1, index=answer_tokens)
    correct_logit, wrong_logit = answer_logits.unbind(dim=-1)
    logit_diff = correct_logit - wrong_logit
    return logit_diff if per_prompt else logit_diff.mean()


def get_logi_diff_direction(model, answer_tokens: Float[Tensor, "batch 2"]):
    answer_residual_directions = model.tokens_to_residual_directions(
        answer_tokens
    )  # [batch 2 d_model]
    correct_residual_directions, incorrect_residual_directions = (
        answer_residual_directions.unbind(dim=1)
    )
    logit_diff_directions = (
        correct_residual_directions - incorrect_residual_directions
    )  # [batch d_model]
    return logit_diff_directions


def residual_stack_to_logit_diff(
    cfg,
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    """
    batch_size = residual_stack.size(-2)
    scaled_final_token_residual_stream = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    if cfg.dtype == "bfloat16":
        average_logit_diff = (
            einops.einsum(
                scaled_final_token_residual_stream.to(torch.bfloat16),
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )
    else:
        average_logit_diff = (
            einops.einsum(
                scaled_final_token_residual_stream,
                logit_diff_directions,
                "... batch d_model, batch d_model -> ...",
            )
            / batch_size
        )
    return average_logit_diff


def test_logit_diff(cfg, cache, logit_diff_directions, original_average_logit_diff):
    final_residual_stream: Float[Tensor, "batch seq d_model"] = cache["resid_post", -1]
    print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[
        :, -1, :
    ]
    average_logit_diff = residual_stack_to_logit_diff(
        cfg, final_token_residual_stream, cache, logit_diff_directions
    )
    print(f"average_logit_diff: {average_logit_diff}")
    print(f"original_average_logit_diff: {original_average_logit_diff}")
    if cfg.dtype != "bfloat16":
        torch.testing.assert_close(
            average_logit_diff, original_average_logit_diff, rtol=0.001, atol=0.001
        )


def residual_logit_diff_acc(cfg, cache, logit_diff_directions):
    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
        cfg, accumulated_residual, cache, logit_diff_directions
    )

    fig = line(
        logit_lens_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800,
        return_fig=True,
    )
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(
        save_path + os.sep + cfg.model_alias + "_logit_diff_accumulated.html"
    )
    pio.write_image(
        fig,
        save_path + os.sep + cfg.model_alias + "_logit_diff_accumulated.png",
        scale=6,
    )


def residual_logit_diff_dec(cfg, cache, logit_diff_directions):
    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    # accumulated_residual has shape (component, batch, d_model)

    logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
        cfg, per_layer_residual, cache, logit_diff_directions
    )

    fig = line(
        logit_lens_logit_diffs,
        hovermode="x unified",
        title="Logit Difference From Accumulated Residual Stream",
        labels={"x": "Layer", "y": "Logit Diff"},
        xaxis_tickvals=labels,
        width=800,
        return_fig=True,
    )
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + os.sep + cfg.model_alias + "_logit_diff_decomposed.html")
    pio.write_image(
        fig,
        save_path + os.sep + cfg.model_alias + "_logit_diff_decomposed.png",
        scale=6,
    )


def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    """
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    """
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


def residual_logit_diff_per_head(cfg, model, cache, logit_diff_directions):
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_head_residual = einops.rearrange(
        per_head_residual,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers,
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(
        cfg, per_head_residual, cache, logit_diff_directions
    )

    fig = imshow(
        per_head_logit_diffs,
        labels={"x": "Head", "y": "Layer"},
        title="Logit Difference From Each Head",
        width=600,
        return_fig=True,
    )
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.write_html(save_path + os.sep + cfg.model_alias + "_logit_diff_per_head.html")
    pio.write_image(
        fig,
        save_path + os.sep + cfg.model_alias + "_logit_diff_per_head.png",
        scale=6,
    )
    return per_head_logit_diffs


def visualize_top_logit_diff_head(cfg, model, cache, per_head_logit_diffs, tokens):
    # save figure
    artifact_dir = cfg.artifact_path()
    save_path = os.path.join(artifact_dir, "logit_diff")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Visualize attention pattern of top k heads
    k = 3
    for head_type in ["Positive", "Negative"]:
        # Get the heads with largest (or smallest) contribution to the logit difference
        top_heads = topk_of_Nd_tensor(
            per_head_logit_diffs * (1 if head_type == "Positive" else -1), k
        )

        # Get all their attention patterns
        attn_patterns_for_important_heads: Float[Tensor, "head q k"] = torch.stack(
            [cache["pattern", layer][:, head].mean(0) for layer, head in top_heads]
        )

        # Display results
        display(HTML(f"<h2>Top {k} {head_type} Logit Attribution Heads</h2>"))

        vis = cv.attention.attention_patterns(
            attention=attn_patterns_for_important_heads,
            tokens=model.to_str_tokens(tokens[:300]),
            attention_head_names=[f"{layer}.{head}" for layer, head in top_heads],
        )

        # save the visualization as .html file
        vis_path = save_path + os.sep + f"vis_layer_top3_heads.html"
        with open(vis_path, "w") as f:
            f.write(vis._repr_html_())


def get_residual_logit_diff(cfg, model, cache, logit_diff_directions):
    residual_logit_diff_acc(cfg, cache, logit_diff_directions)
    residual_logit_diff_dec(cfg, cache, logit_diff_directions)
    per_head_logit_diffs = residual_logit_diff_per_head(
        cfg, model, cache, logit_diff_directions
    )
    return per_head_logit_diffs


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_label: Tuple[str] = ("honest", "lying"),
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_label=contrastive_label,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    # 1. Load model
    model = load_model(cfg)

    # 2. Load data
    print("loading data")
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3. Test prompt
    # test_prompt(cfg, model, dataset_train)

    # 3. Run with cache
    original_logits, cache, prompts, answers, answer_tokens = run_with_cache(
        cfg, model, dataset_train
    )

    original_per_prompt_diff = logits_to_logit_diff(
        original_logits, answer_tokens, per_prompt=True
    )
    print(f"original_per_prompt_diff: {original_per_prompt_diff}")
    original_average_logit_diff = logits_to_logit_diff(
        original_logits, answer_tokens, per_prompt=False
    )
    print(f"original_average_logit_diff: {original_average_logit_diff}")

    logit_diff_directions = get_logi_diff_direction(model, answer_tokens)

    test_logit_diff(cfg, cache, logit_diff_directions, original_average_logit_diff)

    per_head_logit_diffs = get_residual_logit_diff(
        cfg, model, cache, logit_diff_directions
    )

    visualize_top_logit_diff_head(cfg, model, cache, per_head_logit_diffs, prompts[-1])


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_contrastive_generation\n\n")
    print("model_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)

    if args.task_name == "deception":
        contrastive_label = ("honest", "lying")
    elif args.task_name == "jailbreak":
        contrastive_label = ("HHH", args.jailbreak)

    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_label=contrastive_label,
    )
