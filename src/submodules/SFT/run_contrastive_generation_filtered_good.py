import json
import torch
import os
from pipeline.configs.config_generation_filtered_good import Config
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
    # parser.add_argument(
    #     "--model_path_small",
    #     type=str,
    #     required=False,
    #     default="01-ai/Yi-6B-Chat",  # "winnieyangwannan/gemma-2b-it-honest_lying",
    # )
    # parser.add_argument(
    #     "--model_path_big",
    #     type=str,
    #     required=False,
    #     default="meta-llama/Llama-3.3-70B-Instruct",
    # )
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
    parser.add_argument("--task_name", type=str, required=False, default="filtered_good")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


############################################################################################################


def main(model_path_generation, data_dir, save_dir):

    ###################################################################
    # 0. cfg
    model_name_generation = os.path.basename(model_path_generation)

    cfg = Config(
        model_path=model_path_generation,
        model_alias=model_name_generation,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    #############################################################################################################
    # 1. Load dataset
    # data_dir = "D:/Code/deception_and_jailbreak/src/submodules/dataset"
    #

    with open(
        os.path.join(
            data_dir,
            "real_world_dataset_strong_gpt4o_deepseek.json",
        )
    ) as file:
        ds = json.load(file)
    for ii in range(len(ds)):
        if ds[ii]["deceive_answer"] == "Yes":
            ds[ii]["deceive_label"] = 1
            ds[ii]["label"] = 0
            ds[ii]["answer"] = "No"

        else:
            ds[ii]["deceive_label"] = 0
            ds[ii]["label"] = 1
            ds[ii]["answer"] = "Yes"

    ds = pd.DataFrame(ds)
    dataset = {}
    dataset["train"] = ds

    #############################################################################################################
    # 2. Load the model and tokenizer
    model_base = load_model(cfg)

    #############################################################################################################
    contrastive_sft = ContrastiveBase(cfg, model_base, dataset)
    # 3. Prepare dataset --> Generate Messages

    message_honest = contrastive_sft.prepare_dataset_deception(
        model_name_generation, "honest"
    )
    message_lying = contrastive_sft.prepare_dataset_deception(
        model_name_generation, "lying"
    )

    ############################################################################################################
    # # 4. generation
    contrastive_sft.contrastive_generation_with_activation_cache(
        message_honest,
        message_lying,
        train_test="train",
    )

    #############################################################################################################
    # # 5. evaluation
    contrastive_sft.evaluate_deception_real_world()

    # # 6. PCA
    # contrastive_sft.get_contrastive_activation_pca_()
    #
    # print("Get stage statistics:")
    # contrastive_sft.get_stage_statistics()
    # print("Done")


if __name__ == "__main__":
    print("run_contrastive_basic_deception_after_SFT_chinese")

    args = parse_arguments()

    main(
        args.model_path_generation,
        # args.model_path_small,
        # args.model_path_big,
        args.data_dir,
        args.save_dir,
    )
