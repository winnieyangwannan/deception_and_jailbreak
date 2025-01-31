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

########################## ICL ##############################

fig = plt.figure(figsize=(4, 2))
linewidth = 1.3
fontsize = 8
# Yi-6b

########################## ICL ##############################


ax = plt.subplot(1, len(model_path), 1)
# ax.set_title( ["before", "after"], fontsize=20)
plt.plot(
    [0, 1],
    [0.94, 0],
    marker="o",
    color="#A9BBAF",
    markersize=10,
    linewidth=linewidth,  # honest score lying
)
plt.plot(
    [0, 1],
    [0.92, 0.82],
    marker="o",
    color="#C8B8D3",
    markersize=10,
    linewidth=linewidth,
    linestyle="--",  # honest score honest
)
plt.xticks([0, 1], ["Before", "After"], fontsize=fontsize)
plt.yticks(np.arange(0, 1.2, 0.2), fontsize=fontsize)
plt.ylabel("Performance (Accuracy)", fontsize=fontsize)
# no frames
for pos in ["right", "top"]:
    plt.gca().spines[pos].set_visible(False)
plt.gca().spines["bottom"].set_linewidth(linewidth)
plt.gca().spines["left"].set_linewidth(linewidth)
########################## SFT ##############################
ax = plt.subplot(1, len(model_path), 3)
# ax.set_title( ["before", "after"], fontsize=20)
plt.plot(
    [0, 1],
    [0.94, 0.01],
    marker="o",
    color="#A9BBAF",
    markersize=10,
    linewidth=linewidth,  # honest score lying
)
plt.plot(
    [0, 1],
    [0.92, 0.98],
    marker="o",
    color="#C8B8D3",
    markersize=10,
    linewidth=linewidth,
    linestyle="--",  # honest score honest
)


# plt.scatter([0, 1], df["lying_scores"][0:2])
plt.xticks([0, 1], ["Before", "After"], fontsize=fontsize)

plt.yticks(np.arange(0, 1.2, 0.2), fontsize=fontsize)
# plt.ylabel("Performance (Accuracy)", fontsize=fontsize)


# no frames
for pos in ["right", "top"]:
    plt.gca().spines[pos].set_visible(False)

# Set the thickness of the x and y axis lines
plt.gca().spines["bottom"].set_linewidth(linewidth)
plt.gca().spines["left"].set_linewidth(linewidth)
plt.show()


fig.savefig(
    save_path + os.sep + "honest_performance_to_lie_icl_sft.pdf",
    format="pdf",
)
# pio.write_image(
#     fig, save_path + os.sep + "honest_performance_small_big_models.png", scale=6
# )
