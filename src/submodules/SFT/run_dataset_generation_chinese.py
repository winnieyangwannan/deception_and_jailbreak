import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import json
import torch
from datasets import Dataset
import os
from datasets import load_dataset
from pipeline.configs.config_dataset_generation_chinese import Config
import pprint
import argparse
from src.submodules.SFT.contrastive_base_SFT import ContrastiveBase
from src.submodules.evaluations.evaluate_deception_SFT import (
    evaluate_generation_deception_SFT,
)
from src.submodules.model.load_model import load_model

torch.no_grad()
#
# model_path_before = "google/gemma-2b-it"
# model_path_after = "winnieyangwannan/gemma-2b-it-honest_lying"
# model_path_big = "meta-llama/Llama-3.1-70B-Instruct"
# data_dir = ""
# save_dir = ""


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path_generation",
        type=str,
        required=False,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--model_path_small",
        type=str,
        required=False,
        default="01-ai/Yi-6B-Chat",  # "winnieyangwannan/gemma-2b-it-honest_lying",
    )
    parser.add_argument(
        "--model_path_big",
        type=str,
        required=False,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="D:\Code\deception_and_jailbreak\src\submodules\dataset",
    )
    # "D:\Code\deception_and_jailbreak\src\submodules\dataset"
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="D:\Code\deception_and_jailbreak\src\submodules\dataset",
    )
    # D:\Data\deception
    parser.add_argument("--task_name", type=str, required=False, default="deception")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


############################################################################################################


def main(model_path_generation, model_path_small, model_path_big, data_dir, save_dir):

    ###################################################################
    # 0. cfg
    model_name_generation = os.path.basename(model_path_generation)
    model_name_big = os.path.basename(model_path_big)
    model_name_small = os.path.basename(model_path_small)
    model_family_small = model_name_small.split("-")[0].lower()
    model_family_big = model_name_big.split("-")[0].lower()
    cfg = Config(
        model_path=model_path_generation,
        model_alias=model_name_generation,
        model_path_big=model_path_big,
        model_path_small=model_path_small,
        model_name_big=model_name_big,
        model_name_small=model_name_small,
        data_dir=data_dir,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    #############################################################################################################
    # 1. Load dataset
    # data_dir = "D:/Code/deception_and_jailbreak/src/submodules/dataset"
    #
    ds = load_dataset(
        "json",
        data_files=os.path.join(
            cfg.data_dir,
            f"azaria-mitchell-finetune-{model_family_small}-{model_family_big}.json",
        ),
        split="train",
    ).train_test_split(test_size=0.1, seed=0)

    true_ans = [
        ds["train"][ii]["answer"]
        for ii in range(len(ds["train"]))
        if ds["train"][ii]["answer"] == "true"
    ]
    false_ans = [
        ds["train"][ii]["answer"]
        for ii in range(len(ds["train"]))
        if ds["train"][ii]["answer"] == "false"
    ]
    print(f"true statement in training set: {len(true_ans)}")
    print(f"false statement in training set: {len(false_ans)}")
    #############################################################################################################
    # 2. Load the model and tokenizerspli
    model_base = load_model(cfg)

    #############################################################################################################
    contrastive_sft = ContrastiveBase(cfg, model_base, ds)
    # 3. Prepare dataset --> Generate Messages

    train_message = contrastive_sft.prepare_dataset_deception_chinese_translation(
        model_name_generation, ds["train"]
    )

    test_message = contrastive_sft.prepare_dataset_deception_chinese_translation(
        model_name_generation, ds["test"]
    )

    # 4. Tokenization
    inputs_train = contrastive_sft.tokenize_dataset(train_message)
    inputs_test = contrastive_sft.tokenize_dataset(test_message)

    #############################################################################################################
    # 4. generation
    contrastive_sft.generate_batch(
        inputs_train,
        ds["train"],
        train_test="train",
    )
    contrastive_sft.generate_batch(
        inputs_test,
        ds["test"],
        train_test="test",
    )

    #############################################################################################################

    print("Done")


if __name__ == "__main__":
    print("run_contrastive_basic_deception_after_SFT")

    args = parse_arguments()

    main(
        args.model_path_generation,
        args.model_path_small,
        args.model_path_big,
        args.data_dir,
        args.save_dir,
    )
