import os
import json
from collections import defaultdict
from datasets import Dataset
import csv


model_name = "Llama-3.1-8B-Instruct"
model_name = "Llama-3.3-70B-Instruct"

# Open file
file_path = os.path.join(
    "/home/winnieyangwn/Output/truthful_qa",
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
    if example["Score_factual"] == 1 and example["Score_misconception"] == 0:
            data_filetered.append({
                 "Question": example["Question"],
                 "Category": example["Category"],
                 "Best Answer": example["Best Answer"],
                 "Best Incorrect Answer": example["Best Incorrect Answer"],
                 "Source": example["Source"],    
                 "ID": ii
            })

print(f"Total examples after filtering: {len(data_filetered)}")

# Save as huggingface dataset
# fields
save_path = "/home/winnieyangwn/deception_and_jailbreak/src/submodules/truthfulQA/DATA"



# Define the field names (headers)
fieldnames = ["Category", "Question", "Best Answer", "Best Incorrect Answer", "Source", "ID"]

# Writing to CSV
with open(save_path + os.sep + f"{model_name}_truthfulQA.csv", mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Write header row
    writer.writerows(data_filetered)  # Write data rows