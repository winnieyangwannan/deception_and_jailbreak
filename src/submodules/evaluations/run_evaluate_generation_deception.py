from pipeline.configs.config_generation import Config

import json
import os
import argparse
import numpy as np
import sae_lens
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from sklearn.metrics import confusion_matrix
from src.submodules.evaluations.evaluate_deception import (
    evaluate_generation_deception,
    plot_lying_honest_performance,
)


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument("--model_path", type=str, required=True, help="gpt2-small")
    parser.add_argument(
        "--save_dir", type=str, required=False, default="D:\Data\deception"
    )
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=False, default="transformers")
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    temperature: float,
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
    )
    pprint.pp(cfg)
    completions_honest = evaluate_generation_deception(cfg, contrastive_type="honest")
    completions_lying = evaluate_generation_deception(cfg, contrastive_type="lying")
    plot_lying_honest_performance(cfg, completions_honest, completions_lying)


if __name__ == "__main__":
    args = parse_arguments()
    print(sae_lens.__version__)
    print(sae_lens.__version__)
    print("run_pipeline_contrastive_generation\n\n")
    print("model_path")
    print(args.model_path)
    print("save_dir")
    print(args.save_dir)
    print("task_name")
    print(args.task_name)

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
        temperature=args.temperature,
    )
