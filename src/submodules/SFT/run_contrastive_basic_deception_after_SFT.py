import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import json
import torch
from datasets import Dataset
import os
from datasets import load_dataset
from pipeline.configs.config_SFT import Config
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
        "--model_path_after",
        type=str,
        required=False,
        default="winnieyangwannan/Yi-6B-Chat-Llama-3.3-70B-Instruct-honest_lying",
    )
    parser.add_argument(
        "--model_path_before",
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
        default="D:\Data\deception",
    )
    # D:\Data\deception
    parser.add_argument("--task_name", type=str, required=False, default="deception_sft")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


############################################################################################################


def main(model_path_before, model_path_after, model_path_big, data_dir, save_dir):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    ###################################################################
    # 0. cfg
    model_name_small = os.path.basename(model_path_before)
    model_name_big = os.path.basename(model_path_big)
    model_family_small = model_name_small.split("-")[0].lower()
    model_family_big = model_name_big.split("-")[0].lower()
    cfg = Config(
        model_path=model_path_after,
        model_alias=model_name_big,
        model_path_before=model_path_before,
        model_path_after=model_path_after,
        model_path_big=model_path_big,
        model_name_small=model_name_small,
        model_name_big=model_name_big,
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

    train_message_honest = contrastive_sft.prepare_dataset_chat_deception(
        model_name_small, ds["train"], "honest"
    )
    train_message_lying = contrastive_sft.prepare_dataset_chat_deception(
        model_name_small, ds["train"], "lying"
    )
    test_message_honest = contrastive_sft.prepare_dataset_chat_deception(
        model_name_small, ds["test"], "honest"
    )
    test_message_lying = contrastive_sft.prepare_dataset_chat_deception(
        model_name_small, ds["test"], "lying"
    )

    #############################################################################################################
    # 4. generation
    # contrastive_sft.contrastive_generation_with_activation_cache(
    #     train_message_honest,
    #     train_message_lying,
    #     ds["train"],
    #     train_test="train",
    # )
    # contrastive_sft.contrastive_generation_with_activation_cache(
    #     test_message_honest,
    #     test_message_lying,
    #     ds["test"],
    #     train_test="test",
    # )
    #
    # #############################################################################################################
    # # 5. Evaluation
    # contrastive_sft.evaluate_deception()
    #
    # # 6. PCA
    # contrastive_sft.get_contrastive_activation_pca()

    # 6. Stage statistics
    print("Get stage statistics:")
    contrastive_sft.get_stage_statistics(train_test="train")
    contrastive_sft.get_stage_statistics(train_test="test")

    print("Done")


if __name__ == "__main__":
    print("run_contrastive_basic_deception_after_SFT")

    args = parse_arguments()

    main(
        args.model_path_before,
        args.model_path_after,
        args.model_path_big,
        args.data_dir,
        args.save_dir,
    )
