import os
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
import json

import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd

import csv
import math
from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import plotly.express as px


model_path = ["google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct"]


#
# models = [
#     "Qwen(small)",
#     "Qwen(big)",
#     "Yi(small)",
#     "yi(big)",
#     "gemma(small)",
#     "gemma(big)",
#     "llama(small)",
#     "llama(big)",
# ]

# sizes_all = [[7, 57], [6, 34], [7, 27], [8, 70]]
models = ["Gemma", "Llama"]

# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

# size = [1.5, 7, 57, 6, 9, 34, 8, 70, 2, 9, 27]
# data_category = "facts"
Role = ["Honest", "Lying"]
save_path = os.path.join("D:", "\Data", "deception", "figures", "ICLM_submission")
if not os.path.exists(save_path):
    os.makedirs(save_path)


fig = plt.figure(figsize=(4, 2))
linewidth = 1.3
fontsize = 8
performance_honest_all = []
performance_lying_all_before = []
performance_lying_all = []
sem_lying = []
sem_honest = []
sem_lying_before = []

for ii in range(len(model_path)):
    # honest_scores_lying = []
    # honest_scores_honest = []

    model_alias = os.path.basename(model_path[ii])
    if "Llama" in model_alias:
        artifact_path = os.path.join(
            "D:\Data\deception",
            "runs",
            "sft_to_honest",
            "lora_True",
            f"{model_alias}" + "_sft_to_honest_lora_True",
            "100",
            "completions",
        )
        filename = (
            artifact_path
            + os.sep
            + f"{model_alias}_sft_to_honest_lora_True_completions_test.json"
        )
    else:
        artifact_path = os.path.join(
            "D:\Data\deception",
            "runs",
            "sft_to_honest",
            "lora_True",
            f"{model_alias}_honest_lying_sft_to_honest_lora_True",
            "100",
            "completions",
        )
        filename = (
            artifact_path
            + os.sep
            + f"{model_alias}_honest_lying_sft_to_honest_lora_True_completions_test.json"
        )

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    performance_lying_all.append(data["honest_score_lying"])
    performance_lying_all_before.append(data["honest_score_lying_before"])

    # get standard deviation
    honest_score_lying_all = []
    for completion in data["completions"]:
        honest_score_lying_all.append(completion["honest_score_lying"])
    sem = stats.sem(honest_score_lying_all)
    sem_lying.append(sem)

    honest_score_lying_all = []
    for completion in data["completions"]:
        honest_score_lying_all.append(completion["honest_score_lying_before"])
    sem = stats.sem(honest_score_lying_all)
    sem_lying_before.append(sem)

    ######################### honest performance

    honest_scores_honest = [
        data["honest_score_honest_before"],
        data["honest_score_honest"],
    ]
    performance_honest_all.append(data["honest_score_honest_before"])

    # get standard deviation
    honest_score_honest_all = []
    for completion in data["completions"]:
        honest_score_honest_all.append(completion["honest_score_honest"])
    sem = stats.sem(honest_score_honest_all)
    sem_honest.append(sem)


# pio.write_image(
#     fig, save_path + os.sep + "honest_performance_small_big_models.png", scale=6
# )


fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=models,
        y=performance_honest_all,
        name="Honest",
        marker_color="#C8B8D3",
        error_y=dict(
            type="data",  # Use 'data' to specify error values directly
            array=sem_honest,
            visible=True,
        ),
    )
)
fig.add_trace(
    go.Bar(
        x=models,
        y=performance_lying_all_before,
        name="Lying (before sft)",
        marker_color="#A9BBAF",
        error_y=dict(
            type="data",  # Use 'data' to specify error values directly
            array=sem_lying_before,
            visible=True,
        ),
    )
)
fig.add_trace(
    go.Bar(
        x=models,
        y=performance_lying_all,
        name="Lying (after sft)",
        marker_color="#465149",
        error_y=dict(
            type="data",  # Use 'data' to specify error values directly
            array=sem_lying,
            visible=True,
        ),
    )
)

fig.update_layout(
    width=1000,
    height=400,
    xaxis_title=" ",
    yaxis_title="Performance (Accuracy)",
    legend_title="Persona",
    font=dict(
        # family="Times New Roman",
        size=22,
        color="black",
    ),
)
# Here we modify the tickangle of the xaxis, resulting in rotated labels.
# fig.update_layout(barmode="group", xaxis_tickangle=-45)

# save
fig.show()

# fig.write_html(save_path + os.sep + "honest_performance_sft_to_honest.html")
pio.write_image(fig, save_path + os.sep + "honest_performance_sft_to_honest.pdf")
