from pipeline.configs.config_activation_steering import Config
import os
import sys
import random
import json
import pprint
from tqdm import tqdm

from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import einops
import argparse
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
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
from src.submodules.activation_patching.activation_patching import ActivationPatching
from src.submodules.model.load_model import load_model
from src.submodules.activation_steering.contrastive_act_steering import (
    ContrastiveActSteering,
    ContrastiveActSteeringAll,
)
from src.utils.utils import logits_to_ave_logit_diff
from transformer_lens import utils, HookedTransformer, ActivationCache
from rich.table import Table, Column
from rich import print as rprint
from src.utils.plotly_utils import imshow, line, scatter, bar
import plotly.io as pio
import circuitsvis as cv  # for visualize attetnion pattern

from IPython.display import display, HTML  # for visualize attetnion pattern

torch.set_grad_enabled(False)  # save memory


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--source_layer", type=int, required=False, default=0)
    parser.add_argument("--target_layer", type=int, required=False, default=0)
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=False, default="transformers")
    parser.add_argument("--cache_name", type=str, required=False, default="resid_post")

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformers",
    cache_name: str = "resid_pre",
    target_layer: int = 0,
    source_layer: int = 0,
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        task_name=task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        save_dir=save_dir,
        do_sample=do_sample,
        framework=framework,
        cache_name=cache_name,
        source_layer=source_layer,
        target_layer=target_layer,
    )
    pprint.pp(cfg)

    # 1. Load model_base
    model_base = load_model(cfg)
    # model_base.set_use_attn_result(True)
    # model_base.cfg.use_attn_in = True
    # model_base.cfg.use_hook_mlp_in = True

    # 2. Load data
    print("loading data")
    dataset = ContrastiveDatasetDeception(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()
    dataset.construct_contrastive_prompts()

    if cfg.source_layer is None:

        print("Steering layer by layer:")
        for layer in tqdm(range(model_base.cfg.n_layers)):
            print(f"layer:{layer}")
            # Initiate contrastive model base
            print("Initializing model_base base:")
            contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
            # Generate and cache
            print("Run and cache:")
            contrastive_basic.run_and_cache_contrastive_activation()
            act_steer = ContrastiveActSteering(
                cfg, dataset, contrastive_basic, source_layer=layer, target_layer=layer
            )
            steering_vector = act_steer.get_steering_vector()

            dataset = ContrastiveDatasetDeception(cfg)
            dataset.load_and_sample_datasets(train_test="test")
            dataset.process_dataset()
            dataset.construct_contrastive_prompts()
            contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
            act_steer = ContrastiveActSteering(
                cfg, dataset, contrastive_basic, source_layer=layer, target_layer=layer
            )
            act_steer.generate_and_contrastive_activation_steering(steering_vector)
            act_steer.get_contrastive_activation_pca()
            act_steer.evaluate_deception()
            act_steer.get_stage_statistics()

        # summary statistics across layers
        print("Get summary statistics across layers: ")
        act_steer = ContrastiveActSteeringAll(cfg, model_base)
        act_steer.get_honest_score_all_layers()
        act_steer.get_cos_sim_all_layers()

    else:
        contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
        # Generate and cache
        print("Run and cache:")
        contrastive_basic.run_and_cache_contrastive_activation()
        act_steer = ContrastiveActSteering(cfg, dataset, contrastive_basic)
        steering_vector = act_steer.get_steering_vector()

        dataset = ContrastiveDatasetDeception(cfg)
        dataset.load_and_sample_datasets(train_test="test")
        dataset.process_dataset()
        dataset.construct_contrastive_prompts()
        contrastive_basic = ContrastiveBase(cfg, model_base, dataset)
        act_steer = ContrastiveActSteering(cfg, dataset, contrastive_basic)
        act_steer.generate_and_contrastive_activation_steering(steering_vector)
        act_steer.get_contrastive_activation_pca()
        act_steer.evaluate_deception()
        act_steer.get_stage_statistics()

    print("done")


if __name__ == "__main__":
    import time

    start = time.time()
    print("hello")

    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run contrastive activation steering\n\n")
    print("model_base_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    print(args.do_sample)
    print("source_layer")
    print(args.source_layer)
    print("target_layer")
    print(args.target_layer)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
    print("answer_type")
    print(answer_type)
    print("contrastive_type")
    print(contrastive_type)
    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        do_sample=args.do_sample,
        framework=args.framework,
        cache_name=args.cache_name,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
    )
    end = time.time()
    print(end - start)
