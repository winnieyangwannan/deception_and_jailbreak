import os
import argparse

import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd
import json
import csv
import math

from sympy.physics.units import temperature
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


model_path = "01-ai/Yi-6B-Chat"


models = "Yi-6b"

# size = [1.8, 14, 72, 1.5, 7, 57, 6, 34, 6, 9, 34, 7, 13, 70, 8, 70, 2, 7, 2, 9, 27]

# size = [1.5, 7, 57, 6, 9, 34, 8, 70, 2, 9, 27]
data_category = "facts"
Role = ["Honest", "Lying"]
save_path = os.path.join(
    "D:", "\Data", "deception", "figures", "ICLR_revision", "appendix"
)
if not os.path.exists(save_path):
    os.makedirs(save_path)

temperatures = np.arange(0.2, 2.2, 0.2)

lying_scores = []
honest_scores = []

for tt in temperatures:
    tt = round(tt, 2)
    model_alias = os.path.basename(model_path)
    artifact_path = os.path.join(
        "D:\Data\deception", "runs", "activation_pca", model_alias, str(tt)
    )

    # Lying performance
    filename = (
        artifact_path
        + os.sep
        + "completions"
        + os.sep
        + model_alias
        + "_completions_lying.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)
    lying_score = data["honest_score"]
    lying_scores.append(lying_score)

    # honest performance
    filename = (
        artifact_path
        + os.sep
        + "completions"
        + os.sep
        + model_alias
        + "_completions_honest.json"
    )
    with open(filename, "r") as file:
        data = json.load(file)
    honest_score = data["honest_score"]

    honest_scores.append(honest_score)


data_plot = {
    "lying_scores": lying_scores,
    "honest_scores": honest_scores,
    "temperatures": temperatures,
}
df = pd.DataFrame(data_plot)


fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=df["temperatures"],
        y=df["honest_scores"],
        name="honest",
        marker_color="#C8B8D3",
    )
)
fig.add_trace(
    go.Bar(
        x=df["temperatures"], y=df["lying_scores"], name="lying", marker_color="#A9BBAF"
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
    xaxis=dict(
        tickmode="array",
        tickvals=[round(temp, 2) for temp in temperatures],
        ticktext=[round(temp, 2) for temp in temperatures],
    ),
)
fig.update_layout(barmode="group", xaxis_tickangle=-45)

fig.show()


pio.write_image(fig, save_path + os.sep + "honest_performance_temperature.pdf")
pio.write_image(fig, save_path + os.sep + "honest_performance_temperature.png")
