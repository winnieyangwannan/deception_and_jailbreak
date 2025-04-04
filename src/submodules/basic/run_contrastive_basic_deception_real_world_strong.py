from pipeline.configs.config_basic_real_world import Config
import os
import pprint
from typing import Tuple
import argparse

# import sae_lens
import torch
from src.submodules.dataset.old.task_and_dataset_deception_real_world_strong import (
    ContrastiveDatasetDeception,
)
from src.submodules.basic.contrastive_model_base import (
    ContrastiveBase,
)
from src.submodules.model.load_model import load_model

torch.set_grad_enabled(False)  # save memory


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="D:\Data\deception_real_world_strong",
    )
    parser.add_argument(
        "--task_name", type=str, required=False, default="deception_real_world"
    )
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--framework", type=str, required=True, default="transformers")
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


def main(
    model_path: str,
    save_dir: str,
    temperature: int,
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

    # Initiate contrastive model base
    print("Initializing model_base base:")
    contrastive_basic = ContrastiveBase(cfg, model_base, dataset)

    # Generate and cache
    print("Generate and cache:")
    contrastive_basic.generate_and_contrastive_activation_cache()

    # Evaluate generation results
    print("Evaluate generation results:")
    contrastive_basic.evaluate_deception()

    print("Get stage statistics:")
    contrastive_basic.get_stage_statistics()
    # print("Run and cache:")
    contrastive_basic.generate_and_contrastive_activation_cache()

    # Plot pca on the residual stream
    contrastive_basic.get_contrastive_activation_pca()

    print("done")


if __name__ == "__main__":

    args = parse_arguments()
    # print(sae_lens.__version__)
    print("run_contrastive_basic\n\n")
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
    if args.task_name == "deception_real_world":
        contrastive_type = ("honest", "lying")
        answer_type = ("yes", "no")
    elif args.task_name == "jailbreak":
        contrastive_type = ("HHH", args.jailbreak)
        answer_type = ("harmless", "harmful")

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
        do_sample=do_sample,
        framework=args.framework,
        temperature=args.temperature,
    )
