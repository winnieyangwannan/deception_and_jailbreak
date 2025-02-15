import pickle
import os
from sklearn.decomposition import PCA
from datasets import load_dataset
import pandas as pd
import os
from datasets import Dataset
from dataclasses import dataclass
import pprint
import numpy as np
import plotly.io as pio
import pickle
import torch
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly

file_path = os.path.join("D:\Code\deception_and_jailbreak", "results", "basic")
model_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
model_alias = os.path.basename(model_path)


@dataclass
class Config:
    model_path: str
    model_alias: str
    task_name: str = "nnsight_basic"
    contrastive_type: tuple = ("honest", "lying")
    answer_type: tuple = ("yes", "no")
    do_sample: bool = True
    temperature: int = 1.0
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 100  # 100
    batch_size: int = 100  # 256
    max_new_tokens: int = 100  # 100


cfg = Config(
    model_path=model_path,
    model_alias=model_alias,
)
pprint.pp(cfg)


with open(
    file_path + os.sep + "Meta-Llama-3.1-405B-Instruct_activations_train.pkl", "rb"
) as f:
    data = pickle.load(f)


def plot_activation_pca(cfg, activations_positive, activations_negative, dataset):

    activations_all = np.concatenate(
        (activations_positive, activations_negative), axis=1
    )

    n_contrastive_data = activations_negative.shape[1]
    n_layers = activations_negative.shape[0]
    # n_layers = 5
    contrastive_type = cfg.contrastive_type
    answer_type = cfg.answer_type

    labels_all = list(dataset["label"]) + list(dataset["label"])
    answers_all = list(dataset["answer"]) + list(dataset["answer"])

    symbol_sequence = []
    for data in labels_all:
        if int(data) == 1:
            symbol_sequence.append("star")
        else:
            symbol_sequence.append("circle")

    cols = 4
    rows = math.ceil(n_layers / cols)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"layer {n}" for n in range(n_layers)]
    )
    for row in range(rows):
        for ll, layer in enumerate(range(row * 4, row * 4 + 4)):
            # print(f'layer{layer}')
            if layer < n_layers:

                df = {}
                df["label"] = labels_all
                df["pca0"] = activations_all[layer, :, 0]
                df["pca1"] = activations_all[layer, :, 1]
                df["pca2"] = activations_all[layer, :, 2]
                df["answers_all"] = answers_all

                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][:n_contrastive_data],
                        y=df["pca1"][:n_contrastive_data],
                        # z=df['pca2'][:n_contrastive_data],
                        mode="markers",
                        name=contrastive_type[0],
                        showlegend=False,
                        marker=dict(
                            symbol=symbol_sequence[:n_contrastive_data],
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="gold",
                        ),
                        text=df["answers_all"][:n_contrastive_data],
                    ),
                    row=row + 1,
                    col=ll + 1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][n_contrastive_data:],
                        y=df["pca1"][n_contrastive_data:],
                        # z=df['pca2'][n_contrastive_data:],
                        mode="markers",
                        name=contrastive_type[1],
                        showlegend=False,
                        marker=dict(
                            symbol=symbol_sequence[n_contrastive_data:],
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="dodgerblue",
                        ),
                        text=df["answers_all"][n_contrastive_data:],
                    ),
                    row=row + 1,
                    col=ll + 1,
                )
    # legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[0]}_{answer_type[1]}",
            marker_color="gold",
        ),
        row=row + 1,
        col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="star",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[0]}_{answer_type[0]}",
            marker_color="gold",
        ),
        row=row + 1,
        col=ll + 1,
    )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="circle",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[1]}_{answer_type[1]}",
            marker_color="dodgerblue",
        ),
        row=row + 1,
        col=ll + 1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            # z=[None],
            mode="markers",
            marker=dict(
                symbol="star",
                size=5,
                line=dict(width=1, color="darkslategrey"),
                color=df["label"][n_contrastive_data:],
            ),
            name=f"{contrastive_type[1]}_{answer_type[0]}",
            marker_color="dodgerblue",
        ),
        row=row + 1,
        col=ll + 1,
    )
    fig.update_layout(
        height=rows * 250,
        width=cols * 200 + 50,
    )

    fig.update_layout(
        title=dict(
            text=cfg.model_alias,
            font=dict(size=40),
        )
    )
    fig.show()
    # fig.write_html('honest_lying_pca.html')

    return fig


activations_pca_all_honest = data["honest"]
activations_pca_all_lying = data["lying"]
dataset = data["dataset"]

fig = plot_activation_pca(
    cfg, activations_pca_all_honest, activations_pca_all_lying, dataset
)

train_test = "train"
fig.write_image(
    f"{cfg.model_alias}_pca_{train_test}.pdf",
)
