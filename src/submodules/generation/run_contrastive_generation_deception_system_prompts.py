from pipeline.configs.config_generation_system_prompt import Config
import os
import sys
import random
import json
import pprint
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
from src.submodules.model.load_model import load_model
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
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=True, default="transformers")
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--system_prompt_id_s", type=int, required=False, default=0)
    parser.add_argument("--system_prompt_id_e", type=int, required=False, default=9)

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    temperature: float,
    system_prompt_id_s: int,
    system_prompt_id_e: int,
    task_name: str = "deception",
    contrastive_type: Tuple[str] = ("honest", "lying"),
    answer_type: Tuple[str] = ("true", "false"),
    do_sample: bool = False,
    framework: str = "transformer_lens",
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
        temperature=temperature,
        system_prompt_id_s=system_prompt_id_s,
        system_prompt_id_e=system_prompt_id_e,
    )
    pprint.pp(cfg)

    # 1. Load model_base
    model_base = load_model(cfg)

    # 2. Load data
    print("loading data")
    dataset = ContrastiveDatasetDeception(cfg)
    dataset.load_and_sample_datasets(train_test="train")
    dataset.process_dataset()

    for id in range(cfg.system_prompt_id_s, cfg.system_prompt_id_e):
        dataset.construct_contrastive_prompts(system_prompt_id=id)

        # Initiate contrastive model base
        print("Initializing model_base base:")
        cfg.system_prompt_id = id
        contrastive_basic = ContrastiveBase(cfg, model_base, dataset)

        # Generate and cache
        print("Generate:")
        contrastive_basic.generate_and_contrastive()

        # Evaluate generation results
        print("Evaluate generation results:")
        contrastive_basic.evaluate_deception()

        print("done")


if __name__ == "__main__":

    args = parse_arguments()
    print(sae_lens.__version__)
    print("run_contrastive_generation_deception_system_prompts\n\n")
    print("model_base_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    do_sample = True
    print(do_sample)
    print(args.do_sample)

    print("transformer_lens")
    # transformer_lens = False
    # print(transformer_lens)
    print(args.framework)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
        answer_type = ("harmless", "harmful")

    print("answer_type")
    print(answer_type)

    print("contrastive_type")
    print(contrastive_type)

    print("system_prompt_id_s")
    print(args.system_prompt_id_s)
    main(
        model_path=args.model_path,
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        do_sample=do_sample,
        framework=args.framework,
        temperature=args.temperature,
        system_prompt_id_s=args.system_prompt_id_s,
        system_prompt_id_e=args.system_prompt_id_e,
    )
