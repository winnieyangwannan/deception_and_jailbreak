import json
import os
import argparse
import numpy as np
from dataclasses import dataclass

# import sae_lens
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from src.analysis.cos_acc_analysis import (
    plot_cos_sim_and_acc_layer,
)


@dataclass
class Config:
    model_path: str
    model_alias: str
    task_name: str
    save_dir: str
    contrastive_type: str
    answer_type: str
    do_sample: bool
    framework: str
    cache_name: Tuple[str]  # "attn_out" "mlp_out" "resid_pre"
    jailbreak: str
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 100
    n_test: int = 50
    batch_size: int = 5
    max_new_tokens: int = 100
    intervention: str = "no_intervention"
    cache_pos: str = "all"  # "prompt_last_token" or "all" or "prompt_all"
    clean_type: str = "positive"
    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_dir, "runs", "activation_pca", self.model_alias)

    def output_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_dir, "figures", "ICLM_submission", "steer")


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    # parser.add_argument(
    #     "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    # )
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=False)
    parser.add_argument("--framework", type=str, required=False, default="transformers")
    parser.add_argument(
        "--cache_name", nargs="+", type=str, required=False, default="resid_post"
    )
    parser.add_argument(
        "--jailbreak", type=str, required=False, default="evil_confidant"
    )

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
    jailbreak: str = "evil_confidant",
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
        jailbreak=jailbreak,
    )
    pprint.pp(cfg)

    plot_cos_sim_and_acc_layer(cfg)


if __name__ == "__main__":

    args = parse_arguments()

    print("run contrastive activation activation_patching\n\n")

    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)
    print("do_sample")
    print(args.do_sample)

    if args.task_name == "deception":
        contrastive_type = ("honest", "lying")
        answer_type = ("true", "false")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
    print("answer_type")
    print(answer_type)
    print("contrastive_type")
    print(contrastive_type)
    #
    # main(
    #     model_path="google/gemma-2-9b-it",
    #     save_dir=args.save_dir,
    #     task_name=args.task_name,
    #     contrastive_type=contrastive_type,
    #     answer_type=answer_type,
    #     do_sample=args.do_sample,
    #     framework=args.framework,
    #     cache_name=args.cache_name,
    #     jailbreak=args.jailbreak,
    # )

    main(
        model_path="meta-llama/Meta-Llama-3-8B-Instruct",
        save_dir=args.save_dir,
        task_name=args.task_name,
        contrastive_type=contrastive_type,
        answer_type=answer_type,
        do_sample=args.do_sample,
        framework=args.framework,
        cache_name=args.cache_name,
        jailbreak=args.jailbreak,
    )
