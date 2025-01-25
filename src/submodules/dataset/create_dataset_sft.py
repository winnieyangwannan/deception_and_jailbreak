from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
from datasets import Dataset
import os
from datasets import load_dataset, DatasetDict, Dataset
import random
import argparse
import pandas as pd
import torch.nn as nn
import transformers
random.seed(0)
print(transformers.__version__)

def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default=""
    )

    parser.add_argument("--task_name", type=str, required=True, default='sft_to_honest')
    parser.add_argument("--lora", type=bool, required=False, default=True)

    return parser.parse_args()

def main(model_path="google/gemma-2b-it",
         output_dir="",
         task_name="sft_to_lie",
         lora=True):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name = os.path.basename(model_path)


    ###################################################################################
    # 1. Load dataset
    if task_name == "sft_to_lie":
        ds = load_dataset(
             # f"jojoyang/{model_name}_Filter2Honest_TrainTest_240",
            f"jojoyang/{model_name}_Filter2Lie_TrainTest_240"
        )
    elif task_name == "sft_to_honest":
        ds = load_dataset(
             # f"jojoyang/{model_name}_Filter2Honest_TrainTest_240",
            f"jojoyang/{model_name}_Filter2Honest_TrainTest_240"
        )


    true_ans = [ds["train"][ii] for ii in range(len(ds["train"])) if ds["train"][ii]["answer"]=="true"]
    false_ans = [ds["train"][ii]  for ii in range(len(ds["train"])) if ds["train"][ii]["answer"]=="false"]
    print(f"true statement in training set: {len(true_ans)}")
    print(f"false statement in training set: {len(false_ans)}")


    # 2. make sure the class is balanced (true and false)
    if abs(len(true_ans) - len(false_ans))> 100:
        train_ds = []
        if len(true_ans) > len(false_ans):
            dataset_true = random.sample(true_ans,len(false_ans))
            train_ds.append(dataset_true)
            train_ds.append(false_ans)
            train_ds = sum(train_ds, [])
        else:
            dataset_false = random.sample(false_ans, len(true_ans))
            train_ds.append(dataset_false)
            train_ds.append(true_ans)
            train_ds = sum(train_ds, [])
    else:
        train_ds = ds
    # make sure the data is IID
    random.shuffle(train_ds)

    true_ans = [train_ds[ii] for ii in range(len(train_ds)) if train_ds[ii]["answer"]=="true"]
    false_ans = [train_ds[ii]  for ii in range(len(train_ds)) if train_ds[ii]["answer"]=="false"]
    print(f"true statement in training set after ds: {len(true_ans)}")
    print(f"false statement in training set after ds: {len(false_ans)}")

    train_ds = pd.DataFrame(train_ds)
    test_ds = pd.DataFrame(ds["test"])

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_ds),
        "test": Dataset.from_pandas(test_ds),
    })


    # save to huggingface
    # project-name_model-name_task-name
    dataset_dict.push_to_hub(
        f"winnieyangwannan/azaria-mitchell_{model_name}_{task_name}"
    )


if __name__ == "__main__":

    args = parse_arguments()
    main(model_path=args.model_path,
         output_dir=args.output_dir,
         task_name=args.task_name,
         lora=args.lora)

