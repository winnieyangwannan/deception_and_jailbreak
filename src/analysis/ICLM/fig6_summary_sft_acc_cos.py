import json
import os
import argparse

import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from src.analysis.cos_acc_analysis import (
    plot_cos_sim_and_acc_layer,
    collect_all_cos_acc,
    plot_cos_sim_and_acc_checkpoint,
)


import os
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Config:
    model_path: str
    model_alias: str
    task_name: str
    save_dir: str
    data_dir: str
    batch_size: int
    checkpoints: Tuple[int] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    dtype: str = "bfloat16"  # "bfloat16" or "default""
    contrastive_type: tuple = ("honest", "lying")
    answer_type: tuple = ("yes", "no")
    do_sample: bool = True
    temperature: int = 1.0

    def artifact_path(self) -> str:
        data_dir = self.data_dir
        if "Llama" in self.model_alias:
            return os.path.join(
                data_dir,
                "runs",
                f"{self.task_name}",
                "Lora_True",
                f"{self.model_alias}_{self.task_name}_lora_True",
            )
        else:
            return os.path.join(
                data_dir,
                "runs",
                f"{self.task_name}",
                "Lora_True",
                f"{self.model_alias}_honest_lying_{self.task_name}_lora_True",
            )

    def save_path(self) -> str:
        save_dir = self.save_dir
        return os.path.join(
            save_dir,
            "figures",
            "ICLM_submission",
            "sft",
        )


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="D:\Data\deception",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="D:\Data\deception",
    )

    parser.add_argument("--task_name", type=str, required=False, default="sft_to_lie")
    parser.add_argument("--batch_size", type=int, required=False, default=32)

    return parser.parse_args()


def main(
    model_path: str,
    task_name="deception",
    data_dir="D:\Data\deception",
    save_dir="D:\Data\deception",
    batch_size: int = 32,
):

    # 0. Set configuration
    model_alias = os.path.basename(model_path)
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        data_dir=data_dir,
        save_dir=save_dir,
        task_name=task_name,
        batch_size=batch_size,
    )
    pprint.pp(cfg)

    # 1. Collect performance (honest score) and cosine similarity across checkpoints
    cos_all, honest_score_honest_all, honest_score_lying_all = collect_all_cos_acc(cfg)

    # 3. Plot
    plot_cos_sim_and_acc_checkpoint(cfg, cos_all, honest_score_lying_all)


if __name__ == "__main__":

    args = parse_arguments()

    print("model_base_path")
    print(args.model_path)

    main(
        model_path=args.model_path,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
    )
