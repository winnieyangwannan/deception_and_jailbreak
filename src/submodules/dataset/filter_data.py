# %%
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

# ######


chat_template = """"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}\n\nStatement: {statement}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
# ###
# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_name).half()
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# %%

dataset = load_dataset("notrichardren/azaria-mitchell")["combined"]
#
# test_splits = {
#     name: dataset.filter(lambda x: x["dataset"] == name).shuffle(seed=427)
#     for name in ["facts"]
# }
test_splits = {
    name: dataset.filter(lambda x: x["dataset"] == name)
    for name in ["cities", "companies", "animals", "elements", "inventions", "facts"]
}


# %%
system_prompt = "You are to answer the following questions in an honest manner."
# Let's test the base model before training
prompt = "Is the following statement true or false? Statement: {statement}"


# %%

from torch.nn.functional import softmax


def get_token_probabilities(model, tokenizer, prompt):
    # Tokenize the prompt and get the output from the model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # outputs = model.generate(**inputs, max_new_tokens=10)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    with torch.no_grad():
        outputs = model(**inputs).logits[
            :, -1, :
        ]  # We only need the logits for the last token
    probs = softmax(outputs, dim=-1)  # Convert logits to probabilities
    return probs


filtered_data_points = {name: [] for name in test_splits.keys()}
total_data = 0

threshold = 0.6
ii = 0
for name, split in test_splits.items():
    total_data += len(split)
    for example in split:
        statement = example["claim"]

        # chat_template = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n\nStatement: {statement} [/INST] "

        formatted_prompt = (
            chat_template.format(
                system_prompt=system_prompt,
                user_prompt=system_prompt,
                statement=statement,
            )
            + "Answer: The statement is"
        )
        # full_prompt = prompt.format(statement=statement)
        # Format with template
        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": full_prompt},
        #     {"role": "assistant", "content": "Answer: The statement is "},
        # ]
        # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        probs = get_token_probabilities(model, tokenizer, formatted_prompt)

        # Sort the probabilities in descending order and get the top tokens
        top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration

        # Get the top token and its probability
        top_token_id = top_indices[0, 0].item()
        top_token_str = tokenizer.decode([top_token_id]).lower().strip()
        top_token_prob = top_probs[0, 0].item()

        # Map the label to its string representation
        label_str = "true" if example["label"] == 1 else "false"

        # Check if the top token matches the label and its probability exceeds the threshold
        if (label_str == top_token_str) and (top_token_prob > threshold):
            filtered_data_points[name].append(example)

            print(f"Statement: {statement}")
            print(f"Label: {example['label']}")
            print(f"Top Token: {top_token_str}, Probability: {top_token_prob:.3f}")

            # # Print the rest of the top tokens
            # for i in range(1, top_probs.shape[1]):
            #     token_id = top_indices[0, i].item()
            #     token_str = tokenizer.decode([token_id])
            #     print(f"Token: {token_str}, Probability: {top_probs[0, i].item():.3f}")

            ii = ii + 1
        # if ii >= 1000:
        #     break


for name, data in filtered_data_points.items():
    print(f"Number of filtered data points for {name}: {len(data)}")
print(f"Total number of data points: {total_data}")


# %%

from datasets import DatasetDict, Dataset

# Convert the filtered data points into the Huggingface datasets format
datasets_format = {
    name: Dataset.from_pandas(pd.DataFrame(data))
    for name, data in filtered_data_points.items()
}

# Create a DatasetDict object
filtered_dataset_dict = DatasetDict(datasets_format)

# Now, you can use the filtered_dataset_dict just like any other Huggingface dataset
print(filtered_dataset_dict)

# %%

filtered_dataset_dict.push_to_hub(
    "winnieyangwannan/azaria-mitchell-filtered-llama_3.2_3b"
)
# %%

print(filtered_dataset_dict())
