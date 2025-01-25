import json
import torch
import os

# from humanfriendly.terminal import ansi_strip
from pipeline.configs.config_basic_post_sft_lora import Config
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
    parser.add_argument(
        "--task_name", type=str, required=False, default="sft_to_honest"
    )
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


############################################################################################################


def main(model_path, data_dir, save_dir, task_name):

    ###################################################################
    # 0. cfg
    model_name = os.path.basename(model_path)

    cfg = Config(
        model_path=model_path,
        model_alias=model_name,
        save_dir=save_dir,
        task_name=task_name,
    )
    pprint.pp(cfg)

    #############################################################################################################
    # 1. Load dataset
    # todo: this is hard coded now! update !
    model_name_before = "gemma-2-9b-it"
    # dataset_hf = load_dataset("jojoyang/azaria-mitchell_filtered_good")["combined"]
    dataset = load_dataset(
        # f"jojoyang/{model_name}_Filter2Honest_TrainTest_240",
        f"winnieyangwannan/azaria-mitchell_{model_name_before}_{task_name}"
    )

    #
    # # filter out the data where honest_score_honest = 1
    # ds_list = ds["completions"]
    # ds_list_correct = [ ds_list[ii] for ii in range(len(ds_list))
    #                     if ds_list[ii]["honest_score_honest"]==1 and  ds_list[ii]["lying_score_lying"]==1]
    # ds_train = pd.DataFrame( ds_list_correct)
    #
    # dataset = {}
    # dataset["train"] = ds_train

    #############################################################################################################
    # 2. Load the model and tokenizer
    model_base = load_model(cfg)
    # model_base = {}
    #############################################################################################################
    contrastive_sft = ContrastiveBase(cfg, model_base, dataset)
    # # 3. Prepare dataset --> Generate Messages

    message_honest = contrastive_sft.prepare_dataset_chat_deception(
        model_name, "honest"
    )
    message_lying = contrastive_sft.prepare_dataset_chat_deception(model_name, "lying")

    ############################################################################################################
    # # 4. generation
    contrastive_sft.contrastive_generation_with_activation_cache(
        message_honest,
        message_lying,
        train_test="train",
    )

    #############################################################################################################
    # # 5. evaluation
    contrastive_sft.evaluate_deception()

    # # 6. PCA
    contrastive_sft.get_contrastive_activation_pca()
    #
    # print("Get stage statistics:")
    contrastive_sft.get_stage_statistics()
    # print("Done")


if __name__ == "__main__":
    print("run_contrastive_basic_deception_after_SFT_chinese")

    args = parse_arguments()

    main(
        args.model_path,
        # args.model_path_small,
        # args.model_path_big,
        args.data_dir,
        args.save_dir,
        args.task_name,
    )
