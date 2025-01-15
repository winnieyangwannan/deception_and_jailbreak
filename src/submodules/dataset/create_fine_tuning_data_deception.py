import pandas as pd
import os
import json
from datasets import Dataset

model_name_small = "Yi-6B-Chat"
model_name_small_ab = "Yi-6b"
model_name_big = "Llama-3.3-70B-Instruct"
model_name_big_ab = "Llama-33-70b"

completion_path_small = os.path.join(
    "D:\Data\deception",
    "runs",
    "activation_pca",
    model_name_small,
    "1.0",
    "completions",
)
completion_path_big = os.path.join(
    "D:\Data\deception",
    "runs",
    "activation_pca",
    model_name_big,
    "1.0",
    "completions",
)

save_path = os.path.dirname(os.path.realpath(__file__))

## 1. load file
with open(
    os.path.join(completion_path_small, f"{model_name_small}_completions_honest.json"),
    "r",
) as file:
    data_small_honest = json.load(file)

with open(
    os.path.join(completion_path_small, f"{model_name_small}_completions_lying.json"),
    "r",
) as file:
    data_small_lying = json.load(file)

# big model
with open(
    os.path.join(completion_path_big, f"{model_name_big}_completions_honest.json"),
    "r",
) as file:
    data_big_honest = json.load(file)

with open(
    os.path.join(completion_path_big, f"{model_name_big}_completions_lying.json"),
    "r",
) as file:
    data_big_lying = json.load(file)


## 2. Check if for honest data honest score =1 and for lying data, honest score = 0

dataset_finetune = []
dataset_full = []
for ii in range(len(data_small_honest["completions"])):
    if (
        data_small_honest["completions"][ii]["honest_score"] == 1
        and data_big_lying["completions"][ii]["honest_score"] == 0
    ):
        data_full = {}
        data_full["statement"] = data_small_honest["completions"][ii]["statement"]
        data_full["answer"] = data_small_honest["completions"][ii]["answer"]
        data_full["label"] = data_small_honest["completions"][ii]["label"]

        data_full["ID"] = data_small_honest["completions"][ii]["ID"]
        data_full["category"] = data_small_honest["completions"][ii]["category"]

        data_full["response_small_honest"] = data_small_honest["completions"][ii][
            "response"
        ]
        data_full["response_small_lying"] = data_small_lying["completions"][ii][
            "response"
        ]

        data_full["response_big_honest"] = data_big_honest["completions"][ii][
            "response"
        ]
        data_full["response_big_lying"] = data_big_lying["completions"][ii]["response"]

        data_finetune = {}
        data_finetune["statement"] = data_small_honest["completions"][ii]["statement"]
        data_finetune["answer"] = data_small_honest["completions"][ii]["answer"]
        data_finetune["label"] = data_small_honest["completions"][ii]["label"]
        data_finetune["ID"] = data_small_honest["completions"][ii]["ID"]
        data_finetune["category"] = data_small_honest["completions"][ii]["category"]

        data_finetune["response_honest"] = data_small_honest["completions"][ii][
            "response"
        ]
        data_finetune["response_lying"] = data_big_lying["completions"][ii]["response"]

        dataset_full.append(data_full)
        dataset_finetune.append(data_finetune)

true_d = 0
for ii in range(len(dataset_full)):
    if dataset_full[ii]["answer"] == "true":
        true_d += 1
print(f"number of true statements: {true_d}")

false_d = 0
for ii in range(len(dataset_full)):
    if dataset_full[ii]["answer"] == "false":
        false_d += 1
print(f"number of false statements: {false_d}")


# shuffle first
import random
import numpy as np

ind = np.arange(len(dataset_full))
random.shuffle(ind)

dataset_full_shuffle = []
for ii in range(len(dataset_full)):
    dataset_full_shuffle.append(dataset_full[ind[ii]])


true_d = 0
for ii in range(len(dataset_full_shuffle)):
    if dataset_full_shuffle[ii]["answer"] == "true":
        true_d += 1
print(f"number of true statements: {true_d}")

false_d = 0
for ii in range(len(dataset_full_shuffle)):
    if dataset_full_shuffle[ii]["answer"] == "false":
        false_d += 1
print(f"number of false statements: {false_d}")

## downsample
true_d_ds = 0
dataset_ds = []
for ii in range(len(dataset_full_shuffle)):
    if dataset_full_shuffle[ii]["answer"] == "false":
        dataset_ds.append(dataset_full_shuffle[ii])
    elif true_d_ds <= false_d and dataset_full_shuffle[ii]["answer"] == "true":
        dataset_ds.append(dataset_full_shuffle[ii])
        true_d_ds += 1


true_d = 0
for ii in range(len(dataset_ds)):
    if dataset_ds[ii]["answer"] == "true":
        true_d += 1
print(f"number of true statements: {true_d}")

false_d = 0
for ii in range(len(dataset_ds)):
    if dataset_ds[ii]["answer"] == "false":
        false_d += 1
print(f"number of false statements: {false_d}")

## 3 Save
# save locally
save_file = os.path.join(
    save_path,
    f"azaria-mitchell-finetune-{model_name_small_ab}-{model_name_big_ab}.json",
)
with open(save_file, "w") as file:
    json.dump(dataset_full, file, indent=4)


# Convert the filtered data points into the Huggingface datasets format
# datasets_format = Dataset.from_pandas(pd.DataFrame(dataset_finetune))
dataset_hf = Dataset.from_list(dataset_finetune)
dataset_hf.push_to_hub(
    f"winnieyangwannan/azaria-mitchell-finetune-{model_name_small_ab}-{model_name_big_ab}"
)
