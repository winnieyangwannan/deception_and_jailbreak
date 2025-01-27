import json
import torch
import os

# from humanfriendly.terminal import ansi_strip
from pipeline.configs.config_basic_post_sft import Config
import pprint
import argparse
from src.submodules.SFT.contrastive_base_SFT import ContrastiveBase
from src.submodules.model.load_model import load_model
import pandas as pd
from datasets import load_dataset

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
        "--model_path",
        type=str,
        required=False,
        default="01-ai/Yi-6B-Chat",
    )

    # "D:\Code\deception_and_jailbreak\src\submodules\dataset"
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="D:\Data\deception",
    )
    # D:\Data\deception
    parser.add_argument("--task_name", type=str, required=True, default="sft_to_honest")
    parser.add_argument("--checkpoint", type=int, required=False, default=None)

    return parser.parse_args()


############################################################################################################


def main(model_path, checkpoint, save_dir, task_name):

    ###################################################################
    # 0. cfg

    model_alias = os.path.basename(model_path)
    model_alias_before = model_alias.split("_")[0]
    if "gemma" in model_alias_before:
        model_path_before = "google/" + model_alias_before
    elif "Llama" in model_alias_before:
        model_path_before = "meta-llama/" + model_alias_before
    elif "Qwen" in model_alias_before:
        model_path_before = "Qwen/" + model_alias_before
    elif "Yi" in model_alias_before:
        model_path_before = "01-ai/" + model_alias_before

    lora = model_path.split("_")[-1]

    if lora == "True":
        lora = True
    else:
        lora = False

    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        model_path_before=model_path_before,
        model_alias_before=model_alias_before,
        save_dir=save_dir,
        task_name=task_name,
        lora=lora,
        checkpoint=checkpoint,
    )
    pprint.pp(cfg)

    #############################################################################################################

    #############################################################################################################
    # 2. Load the model and tokenizer
    model_base = load_model(cfg)
    # model_base = {}

    # 1. Load dataset

    # dataset_hf = load_dataset("jojoyang/azaria-mitchell_filtered_good")["combined"]
    dataset = load_dataset(
        f"winnieyangwannan/azaria-mitchell_{model_alias_before}_{task_name}"
    )
    #############################################################################################################
    contrastive_sft = ContrastiveBase(cfg, model_base, dataset)
    # # 3. Prepare dataset --> Generate Messages

    message_honest_train = contrastive_sft.prepare_dataset_chat_deception(
        model_alias, "honest", train_test="train"
    )
    message_lying_train = contrastive_sft.prepare_dataset_chat_deception(
        model_alias, "lying", train_test="train"
    )

    message_honest_test = contrastive_sft.prepare_dataset_chat_deception(
        model_alias, "honest", train_test="test"
    )
    message_lying_test = contrastive_sft.prepare_dataset_chat_deception(
        model_alias, "lying", train_test="test"
    )

    ############################################################################################################
    # # 4. generation
    contrastive_sft.contrastive_generation_with_activation_cache(
        message_honest_train,
        message_lying_train,
        train_test="train",
    )
    contrastive_sft.contrastive_generation_with_activation_cache(
        message_honest_test,
        message_lying_test,
        train_test="test",
    )
    #############################################################################################################
    # # 5. evaluation
    contrastive_sft.evaluate_deception()

    # # 6. PCA
    contrastive_sft.get_contrastive_activation_pca()
    #
    print("Get stage statistics:")
    # contrastive_sft.get_stage_statistics(train_test="train")
    contrastive_sft.get_stage_statistics(train_test="test")

    print("Done")


if __name__ == "__main__":
    print("run_contrastive_basic_post_sft_lora")

    args = parse_arguments()

    main(
        args.model_path,
        args.checkpoint,
        args.save_dir,
        args.task_name,
    )
