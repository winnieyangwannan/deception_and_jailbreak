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


model_path = [
    ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2-57B-A14B-Instruct"],
    ["01-ai/Yi-6B-Chat", "01-ai/Yi-1.5-34B-Chat"],
    ["google/gemma-7b-it", "google/gemma-2-27b-it"],
    ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.3-70B-Instruct"],
]
titles = ["Qwen", "Yi", "Gemma", "Llama"]

sizes_all = [[7, 57], [6, 34], [7, 27], [8, 70]]

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
for ii in range(len(model_path)):
    honest_scores_lying = []
    honest_scores_honest = []
    sem_lying = []
    sem_honest = []
    for jj in range(len(model_path[ii])):
        print(model_path[ii][jj])
        model_alias = os.path.basename(model_path[ii][jj])
        artifact_path = os.path.join(
            "D:\Data\deception", "runs", "real_world", model_alias
        )

        # Lying performance
        filename = (
            artifact_path
            + os.sep
            + "completions"
            + os.sep
            + model_alias
            + "_completions_train.json"
        )
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        honest_score_lying = data["honest_score_lying"]
        honest_scores_lying.append(honest_score_lying)

        # get standard deviation
        honest_score_lying_all = []
        for completion in data["completions"]:
            honest_score_lying_all.append(completion["honest_score_lying"])
        sem = stats.sem(honest_score_lying_all)
        sem_lying.append(sem)

        honest_score_honest = data["honest_score_honest"]
        honest_scores_honest.append(honest_score_honest)

        # get standard deviation
        honest_score_honest_all = []
        for completion in data["completions"]:
            honest_score_honest_all.append(completion["honest_score_honest"])
        sem = stats.sem(honest_score_honest_all)
        sem_honest.append(sem)

    ax = plt.subplot(1, len(model_path), ii + 1)
    ax.set_title(titles[ii], fontsize=20)
    plt.plot(
        [0, 1],
        honest_scores_lying[0:2],
        marker="o",
        color="#A9BBAF",
        markersize=10,
        linewidth=linewidth,
    )
    plt.plot(
        [0, 1],
        honest_scores_honest[0:2],
        marker="o",
        color="#C8B8D3",
        markersize=10,
        linewidth=linewidth,
        linestyle="--",
    )
    # plt.scatter([0, 1], df["lying_scores"][0:2])
    plt.xticks([0, 1], sizes_all[ii], fontsize=fontsize)
    if ii == 0:
        plt.yticks(np.arange(0, 1.2, 0.2), fontsize=fontsize)
        plt.ylabel("Performance (Accuracy)", fontsize=fontsize)

    else:
        plt.yticks([])

    # no frames

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)

    # Set the thickness of the x and y axis lines
    plt.gca().spines["bottom"].set_linewidth(linewidth)
    plt.gca().spines["left"].set_linewidth(linewidth)
plt.show()


fig.savefig(
    save_path + os.sep + "honest_performance_real_world_small_big_models.pdf",
    format="pdf",
)
# pio.write_image(
#     fig, save_path + os.sep + "honest_performance_small_big_models.png", scale=6
# )
