import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd
import json
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
import scipy.stats as stats


model_path = [
    # "Qwen/Qwen-1_8B-Chat",
    # "Qwen/Qwen-14B-Chat",
    # "Qwen/Qwen2-1.5B-Instruct",
    # "Qwen/Qwen2-7B-Instruct",
    # "01-ai/Yi-6B-Chat",
    # "01-ai/Yi-34B-Chat",
    # "01-ai/Yi-1.5-6B-Chat",
    # "01-ai/Yi-1.5-9B-Chat",
    # "meta-llama/Llama-2-7b-chat-hf",
    # "meta-llama/Llama-2-13b-chat-hf",
    # "meta-llama/Llama-2-70b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    # "Meta-Llama-3-70B-Instruct",
    # "google/gemma-2b-it",
    # "google/gemma-7b-it",
    # "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    # "01-ai/Yi-1.5-34B-Chat",
    # "Qwen/Qwen2-57B-A14B-Instruct",
    # "google/gemma-2-27b-it",
]


models = [
    # "Qwen-1.8b",
    # "Qwen-14b",
    # "Qwen-2-1.5b",
    # "Qwen-2-7b",
    # "Yi-1-6b",
    # "Yi-1-34b",
    # "Yi-1.5-6b",
    # "Yi-1.5-9b",
    # "llama-2-7b",
    # "llama-2-13b",
    # "llama-2-70b",
    "Llama",
    # "llama-3-70b",
    # "gemma-1-2b",
    # "gemma-1-7b",
    # "gemma-2-2b",
    "Gemma",
    # "Yi-1.5-34b",
    # "Qwen-2-57b",
    # "gemma-2-27b",
]

# size = [1.8, 14, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]
size = [
    8,
    9,
]

Role = ["Honest", "Lying"]
metric = ["performance_lying", "cosine_sim"] * len(model_path)
save_path = os.path.join("D:", "\Data", "deception", "figures", "ICLM_submission")
if not os.path.exists(save_path):
    os.makedirs(save_path)

performance_lying_all = []
performance_honest_all = []
performance_lying_all_steer = []

sem_lying = []
sem_honest = []
sem_lying_steer = []

for mm in model_path:
    print(mm)

    # lying with steering
    model_alias = os.path.basename(mm)
    load_path = os.path.join(
        "D:\Data\deception",
        "runs",
        "activation_pca",
        model_alias,
        "activation_steering",
        "positive_addition",
        "performance",
    )
    filename = load_path + os.sep + model_alias + "_honest_score_all_layers.pkl"
    with open(filename, "br") as file:
        data = pickle.load(file)
    performance_lying = data["score_negative_all"]
    performance_lying_all_steer.append(np.max(performance_lying))
    max_layer = np.argmax(performance_lying)

    # load maximum layer --> to estimate sem for error bar
    load_path = os.path.join(
        "D:\Data\deception",
        "runs",
        "activation_pca",
        model_alias,
        "activation_steering",
        "positive_addition",
        "completions",
    )
    filename = (
        load_path
        + os.sep
        + model_alias
        + f"_completions_source_{max_layer}_target_{max_layer}_lying.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)

    honest_score_al = []
    for completion in data["completions"]:
        honest_score_al.append(completion["honest_score"])
    sem = stats.sem(honest_score_al)
    sem_lying_steer.append(sem)

    # lying before steering
    model_alias = os.path.basename(mm)
    load_path = os.path.join(
        "D:\Data\deception",
        "runs",
        "activation_pca",
        model_alias,
        "completions",
    )
    filename = load_path + os.sep + model_alias + "_completions_lying.json"
    with open(filename, "r") as file:
        data = json.load(file)

    performance_lying = data["honest_score"]
    performance_lying_all.append(performance_lying)

    honest_score_al = []
    for completion in data["completions"]:
        honest_score_al.append(completion["honest_score"])
    sem = stats.sem(honest_score_al)
    sem_lying.append(sem)

    # honest
    filename = load_path + os.sep + model_alias + "_completions_honest.json"
    with open(filename, "r") as file:
        data = json.load(file)
    performance_honest = data["honest_score"]
    performance_honest_all.append(performance_honest)

    honest_score_al = []
    for completion in data["completions"]:
        honest_score_al.append(completion["honest_score"])
    sem = stats.sem(honest_score_al)
    sem_honest.append(sem)


data_plot = {
    "performance_lying_all": performance_lying_all,
    "performance_honest_all": performance_honest_all,
    "performance_lying_all_steer": performance_lying_all_steer,
    "model_name": models,
    "size": size,
    "color": ["salmon"] * len(models),
}
df = pd.DataFrame(data_plot)


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
        y=performance_lying_all,
        name="Lying (before steering)",
        marker_color="#A9BBAF",
        error_y=dict(
            type="data",  # Use 'data' to specify error values directly
            array=sem_lying,
            visible=True,
        ),
    )
)
fig.add_trace(
    go.Bar(
        x=models,
        y=performance_lying_all_steer,
        name="Lying (after steering)",
        marker_color="#465149",
        error_y=dict(
            type="data",  # Use 'data' to specify error values directly
            array=sem_lying_steer,
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
# fig.write_html(save_path + os.sep + "honest_performance_steering_to_honest.html")
# pio.write_image(
#     fig, save_path + os.sep + "honest_performance_steering_to_honest.png", scale=6
# )
pio.write_image(fig, save_path + os.sep + "honest_performance_steering_to_honest.pdf")
