import torch
import os
import math
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import pickle
import json



def get_contrastive_activation_pca_(cfg, activations, save_name=None, split="train"):
    # 1. Get PCA and plot
    fig, activations_pca_all = get_pca_and_plot(
        cfg,
        activations["activations_positive"],
        activations["activations_negative"],
    )

    # 2. Save plot
    pca_path = os.path.join(cfg.artifact_path(), "pca")
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)
    fig.write_html(
        pca_path + os.sep + f"{cfg.model_name}_pca_{split}.html"
    )
    pio.write_image(
        fig,
        pca_path + os.sep + f"{cfg.model_name}_pca_{split}.pdf",
    )
    print("PCA Saved...")



def get_pca_and_plot(cfg, activations_positive, activations_negative):


    print(f"activations_positive:{activations_positive.shape}")
    print(f"activations_positive:{activations_positive.shape}")

    activations_all: Float[Tensor, "n_layers n_samples  d_model"] = torch.cat(
        (activations_positive, activations_negative), dim=1
    )



    n_contrastive_data = activations_negative.shape[1]
    n_layers = activations_negative.shape[0]
    contrastive_type = cfg.contrastive_type

    print(f"n_contrastive_data: {n_contrastive_data}")
    print(f"n_layers: {n_layers}")
    print(f"contrastive_type: {contrastive_type}")



    cols = 7
    rows = math.ceil(n_layers / cols)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"layer {n}" for n in range(n_layers)]
    )
    n_components = 3
    pca = PCA(n_components=n_components)
    activations_pca_all = np.zeros((n_layers, n_contrastive_data * 2, n_components))
    for row in range(rows):
        for ll, layer in enumerate(range(row * cols, row * cols + cols)):
            # print(f'layer{layer}')
            if layer < n_layers:
                activations_pca = pca.fit_transform(
                    activations_all[layer, :, :].float().detach().cpu().numpy()
                )
                activations_pca_all[layer, :, :] = activations_pca
                df = {}
                df["pca0"] = activations_pca[:, 0]
                df["pca1"] = activations_pca[:, 1]
                df["pca2"] = activations_pca[:, 2]

                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][:n_contrastive_data],
                        y=df["pca1"][:n_contrastive_data],
                        # z=df['pca2'][:n_contrastive_data],
                        mode="markers",
                        name=contrastive_type[0],
                        showlegend=False,
                        marker=dict(
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="teal",
                        ),
                        text=contrastive_type[0],
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
                            size=8,
                            line=dict(width=1, color="darkslategrey"),
                            color="palevioletred",
                        ),
                        text=contrastive_type[1],
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
                size=5,
                line=dict(width=1, color="darkslategrey"),
            ),
            name=f"{contrastive_type[0]}",
            marker_color="teal",
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
            ),
            name=f"{contrastive_type[1]}",
            marker_color="palevioletred",
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
            text=cfg.model_name,
            font=dict(size=40),
        )
    )
    # fig.show()
    # fig.write_html('honest_lying_pca.html')
    print("PCA Generated...")
    return fig, activations_pca_all