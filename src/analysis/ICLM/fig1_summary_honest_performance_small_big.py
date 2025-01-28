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
    ["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-57B-A14B-Instruct"],
    ["01-ai/Yi-6B-Chat", "01-ai/Yi-1.5-34B-Chat"],
    ["google/gemma-7b-it", "google/gemma-2-27b-it"],
    ["Meta-Llama-3-8B-Instruct", "Llama-3.3-70B-Instruct"],
]

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

sizes_all = [[7, 57], [6, 34], [7, 27], [8, 70]]

# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

# size = [1.5, 7, 57, 6, 9, 34, 8, 70, 2, 9, 27]
# data_category = "facts"
Role = ["Honest", "Lying"]
save_path = os.path.join("D:", "\Data", "deception", "figures", "ICLM_submission")
if not os.path.exists(save_path):
    os.makedirs(save_path)


fig = plt.figure()

for ii in range(len(model_path)):
    honest_scores_lying = []
    honest_scores_honest = []
    sem_lying = []
    sem_honest = []
    for jj in range(len(model_path[ii])):
        print(model_path[ii][jj])
        model_alias = os.path.basename(model_path[ii][jj])
        artifact_path = os.path.join(
            "D:\Data\deception", "runs", "activation_pca", model_alias
        )

        # Lying performance
        filename = (
            artifact_path
            + os.sep
            + "1.0"
            + os.sep
            + "completions"
            + os.sep
            + model_alias
            + "_completions_lying.json"
        )
        with open(filename, "r") as file:
            data = json.load(file)

        honest_score_lying = data["honest_score"]
        honest_scores_lying.append(honest_score_lying)

        # get standard deviation
        honest_score_lying_all = []
        for completion in data["completions"]:
            honest_score_lying_all.append(completion["honest_score"])
        sem = stats.sem(honest_score_lying_all)
        sem_lying.append(sem)

        # honest performance
        filename = (
            artifact_path
            + os.sep
            + "1.0"
            + os.sep
            + "completions"
            + os.sep
            + model_alias
            + "_completions_honest.json"
        )
        with open(filename, "r") as file:
            data = json.load(file)
        honest_score_honest = data["honest_score"]
        honest_scores_honest.append(honest_score_honest)

        # get standard deviation
        honest_score_honest_all = []
        for completion in data["completions"]:
            honest_score_honest_all.append(completion["honest_score"])
        sem = stats.sem(honest_score_honest_all)
        sem_honest.append(sem)

    plt.subplot(1, len(model_path), ii + 1)

    plt.plot(
        [0, 1],
        honest_scores_lying[0:2],
        marker="o",
        color="#A9BBAF",
        markersize=10,
        linewidth=2,
    )
    plt.plot(
        [0, 1],
        honest_scores_honest[0:2],
        marker="o",
        color="#C8B8D3",
        markersize=10,
        linewidth=2,
        linestyle="--",
    )
    # plt.scatter([0, 1], df["lying_scores"][0:2])
    plt.xticks([0, 1], sizes_all[ii], fontsize=16)
    if ii == 0:
        plt.yticks(np.arange(0, 1.2, 0.2), fontsize=16)
        plt.ylabel("Performance (Accuracy)", fontsize=20)

    else:
        plt.yticks([])

    # no frames

    for pos in ["right", "top"]:
        plt.gca().spines[pos].set_visible(False)

    # Set the thickness of the x and y axis lines
    plt.gca().spines["bottom"].set_linewidth(2)
    plt.gca().spines["left"].set_linewidth(2)
plt.show()


fig.savefig(
    save_path + os.sep + "honest_performance_small_big_models.pdf", format="pdf"
)
# pio.write_image(
#     fig, save_path + os.sep + "honest_performance_small_big_models.png", scale=6
# )
