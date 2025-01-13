import pandas as pd
import os
import json
from datasets import Dataset

model_name_small = "gemma-2b-it"
model_name_big = "Llama-3.1-70B-Instruct"

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
        data_full["answer"] = data_small_honest["completions"][ii]["label"]
        data_full["ID"] = data_small_honest["completions"][ii]["ID"]
        data_full["category"] = data_small_honest["completions"][ii]["category"]

        if data_small_honest["completions"][ii]["label"] == "false":
            data_full["label"] = 0
        elif data_small_honest["completions"][ii]["label"] == "true":
            data_full["label"] = 1

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
        data_finetune["answer"] = data_small_honest["completions"][ii]["label"]
        data_finetune["ID"] = data_small_honest["completions"][ii]["ID"]
        data_finetune["category"] = data_small_honest["completions"][ii]["category"]

        if data_small_honest["completions"][ii]["label"] == "false":
            data_finetune["label"] = 0
        else:
            data_finetune["label"] = 1

        data_finetune["response_honest"] = data_small_honest["completions"][ii][
            "response"
        ]
        data_finetune["response_lying"] = data_big_lying["completions"][ii]["response"]

        dataset_full.append(data_full)
        dataset_finetune.append(data_finetune)


## 3 Save
# save locally
save_file = os.path.join(save_path, f"azaria-mitchell-finetune-gemma-2b-llama-70b.json")
with open(save_file, "w") as file:
    json.dump(dataset_full, file, indent=4)


# Convert the filtered data points into the Huggingface datasets format
datasets_format = Dataset.from_pandas(pd.DataFrame(dataset_finetune))
dataset_hf = Dataset.from_list(dataset_finetune)
dataset_hf.push_to_hub(f"winnieyangwannan/azaria-mitchell-finetune")
