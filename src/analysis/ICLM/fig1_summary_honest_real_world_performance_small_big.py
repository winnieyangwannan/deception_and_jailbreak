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


model_path = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-57B-A14B-Instruct",
    "01-ai/Yi-6B-Chat",
    "01-ai/Yi-1.5-34B-Chat",
    "google/gemma-7b-it",
    "google/gemma-2-27b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.3-70B-Instruct",
]


models = [
    "Qwen(7b)",
    "Qwen(57b)",
    "Yi(6b)",
    "yi(34b)",
    "gemma(7b)",
    "gemma(27b)",
    "llama(8b)",
    "llama(70)",
]

# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

# size = [1.5, 7, 57, 6, 9, 34, 8, 70, 2, 9, 27]
# data_category = "facts"
Role = ["Honest", "Lying"]
save_path = os.path.join("D:", "\Data", "deception", "figures", "ICLM_submission")
if not os.path.exists(save_path):
    os.makedirs(save_path)

lying_scores = []
honest_scores = []

for mm in model_path:
    print(mm)
    model_alias = os.path.basename(mm)
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
    with open(filename, "r",  encoding="utf-8") as file:
        data = json.load(file)
    lying_score = data["honest_score_lying"]
    honest_score = data["honest_score_honest"]

    honest_scores.append(honest_score)
    lying_scores.append(lying_score)


data_plot = {
    "lying_scores": lying_scores,
    "honest_scores": honest_scores,
    "model_name": models,
}
df = pd.DataFrame(data_plot)


fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=df["model_name"],
        y=df["honest_scores"],
        name="honest",
        marker_color="#C8B8D3",
    )
)
fig.add_trace(
    go.Bar(
        x=df["model_name"], y=df["lying_scores"], name="lying", marker_color="#A9BBAF"
    )
)

fig.update_layout(
    height=500,
    width=1000,
    yaxis_title="Performance (Accuracy)",
    font=dict(
        size=20,
        color="black",
    ),
)
fig.update_layout(barmode="group", xaxis_tickangle=-45)

fig.show()


pio.write_image(fig, save_path + os.sep + "honest_performance_real_world_all_models.pdf")
pio.write_image(fig, save_path + os.sep + "honest_performance_real_world_all_models.png", scale=6)
