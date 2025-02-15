import json
import torch
import os
from pipeline.configs.config_basic_nnsight import Config
from nnsight import CONFIG

# from humanfriendly.terminal import ansi_strip
import pprint
import argparse
from src.submodules.NNsight.contrastive_base_nnsight import ContrastiveBase
from src.submodules.model.load_model import load_model
import pandas as pd
from datasets import load_dataset
from nnsight import LanguageModel


torch.no_grad()


def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="meta-llama/Meta-Llama-3.1-405B-Instruct",  # "meta-llama/Meta-Llama-3.1-405B-Instruct",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="D:\Data\deception",
    )
    parser.add_argument(
        "--nnsight_key",
        type=str,
        required=False,
        default="",
    )

    return parser.parse_args()


def main(model_path, save_dir):

    model_alias = os.path.basename(model_path)

    # 0. config
    cfg = Config(
        model_path=model_path,
        model_alias=model_alias,
        save_dir=save_dir,
    )
    pprint.pp(cfg)

    # 1. load dataset
    dataset = load_dataset(
        f"winnieyangwannan/azaria-mitchell_gemma-2-9b-it_sft_to_honest"
    )

    # 1.2. get 100 facts
    df = pd.DataFrame(dataset["train"])
    facts = df[df["category"] == "facts"]
    if len(facts) > 100:
        dataset_fact = facts.sample(n=100, random_state=seed)

    # 2. Load model

    model = LanguageModel(model_path, allow_multiple=True)
    # tokenizer = model.tokenizer

    # 3. Process dataset
    contrastive_base = ContrastiveBase(cfg, model, dataset_fact)
    prompt_honest = contrastive_base.prepare_dataset_chat_deception(
        model_alias, "honest"
    )
    prompt_lying = contrastive_base.prepare_dataset_chat_deception(model_alias, "lying")

    # 4. run nnsight, do pca and save
    with model.trace(remote=True) as runner:
        with runner.invoke(prompt_honest) as invoker:
            # Save the model's hidden states
            hidden_states = model.model.layers[-2].output[0].save()
    hidden_states.shape
    # 5. do pca


if __name__ == "__main__":
    print("run_nnsight_basic")

    args = parse_arguments()
    seed = 0

    CONFIG.set_default_api_key(args.nnsight_key)
    main(args.model_path, args.save_dir)
