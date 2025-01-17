import json
import torch
import os
from pipeline.configs.config_SFT_chinese import Config
import pprint
import argparse
from src.submodules.SFT.contrastive_base_SFT import ContrastiveBase
from src.submodules.model.load_model import load_model
import pandas as pd

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
        default="01-ai/Yi-6B-Chat",
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
    # todo: change this
    with open(
        os.path.join(
            cfg.data_dir,
            f"{model_name_small}-{model_name_big}_dataset_chinese_test.json",
        )
    ) as file:
        ds_train = json.load(file)

    with open(
        os.path.join(
            cfg.data_dir,
            f"{model_name_small}-{model_name_big}_dataset_chinese_test.json",
        )
    ) as file:
        ds_test = json.load(file)
    ds = {}
    ds["train"] = pd.DataFrame(ds_train)
    ds["test"] = pd.DataFrame(ds_test)

    # aa = pd.DataFrame(ds_train)

    #############################################################################################################
    # 2. Load the model and tokenizer
    model_base = load_model(cfg)

    #############################################################################################################
    contrastive_sft = ContrastiveBase(cfg, model_base, ds)
    # 3. Prepare dataset --> Generate Messages

    train_message_honest = contrastive_sft.prepare_dataset_deception_chinese_response(
        model_name_small, ds["train"], "honest"
    )
    train_message_lying = contrastive_sft.prepare_dataset_deception_chinese_response(
        model_name_small, ds["train"], "lying"
    )
    test_message_honest = contrastive_sft.prepare_dataset_deception_chinese_response(
        model_name_small, ds["test"], "honest"
    )
    test_message_lying = contrastive_sft.prepare_dataset_deception_chinese_response(
        model_name_small, ds["test"], "lying"
    )

    #############################################################################################################
    # 4. generation
    contrastive_sft.contrastive_generation_with_activation_cache(
        train_message_honest,
        train_message_lying,
        ds["train"],
        train_test="train",
        generation_type="SFT_chinese",
    )
    contrastive_sft.contrastive_generation_with_activation_cache(
        test_message_honest,
        test_message_lying,
        ds["test"],
        train_test="test",
        generation_type="SFT_chinese",
    )

    #############################################################################################################
    # # 5. evaluation
    contrastive_sft.evaluate_deception()

    # 6. PCA
    contrastive_sft.get_contrastive_activation_pca()

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
