import os
import json
from collections import defaultdict
from datasets import Dataset
import csv


model_name = "Llama-3.1-8B-Instruct"
model_name = "Llama-3.3-70B-Instruct"

task_name = "sandbag"
category = "wmdp-bio"



save_path = "/home/winnieyangwn/deception_and_jailbreak/src/submodules/truthfulQA/DATA"

# Open file
file_path = os.path.join(
    f"/home/winnieyangwn/Output",
    task_name,
    category,
    model_name,
    "completions",
    f"{model_name}_completions_train.json",
)
with open(file_path) as file:
    data = json.load(file)


print(f"Total examples before filtering: {len(data['completions'])}")
# filter
# Criteria: 1. score_factual = 1, score_misconception = 0
data_filetered =  []
for ii, example in enumerate(data["completions"]):
    if example["score_control"] == 1 and example["score_sandbag"] == 0:
            data_filetered.append({
                 "question": example["question"],
                 "correct_choice": example["correct_choice"],
                 "ID": ii,
                 "correct_answer": example["correct_answer"],
                 "wrong_answer": example["wrong_answer"],
            })


# Define the field names (headers)
fieldnames = ["question", "answer", "correct_choice", "correct_answer", "wrong_answer", "ID"]

# Writing to CSV
with open(save_path + os.sep + f"{model_name}_{task_name}_{category}.csv", mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write header row
    writer.writerows(data_filetered)  # Write data rows