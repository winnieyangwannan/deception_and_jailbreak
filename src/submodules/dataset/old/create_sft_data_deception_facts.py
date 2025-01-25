import pandas as pd
import os
import json
from datasets import Dataset

model_name_small = "Yi-6B-Chat"
model_name_big = "Llama-3.3-70B-Instruct"
model_family_small = model_name_small.split("-")[0].lower()
model_family_big = model_name_big.split("-")[0].lower()

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



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# filter out only facts dat
facts = [dataset_full[ii] for ii in range(len(dataset_full)) if dataset_full[ii]["category"] == "facts"]

len(facts)


true_d = 0 # 282
for ii in range(len(facts)):
    if facts[ii]["answer"] == "true":
        true_d += 1
print(f"number of true statements: {true_d}")

false_d = 0 # 217
for ii in range(len(facts)):
    if facts[ii]["answer"] == "false":
        false_d += 1
print(f"number of false statements: {false_d}")

# save locally
save_file = os.path.join(
    save_path,
    f"azaria-mitchell-finetune-{model_family_small}-{model_family_big}-facts.json",
)
with open(save_file, "w") as file:
    json.dump(dataset_ds, file, indent=4)


# save to huggingface
# Convert the filtered data points into the Huggingface datasets format
# datasets_format = Dataset.from_pandas(pd.DataFrame(dataset_finetune))
dataset_hf = Dataset.from_list(dataset_ds)
dataset_hf.train_test_split(test_size=0.1, seed=0)
dataset_hf.push_to_hub(
    f"winnieyangwannan/azaria-mitchell-finetune-{model_family_small}-{model_family_big}-facts"
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

