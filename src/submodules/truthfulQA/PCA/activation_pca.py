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

import umap





def get_contrastive_activation_dim_reduction(cfg, activations, save_name=None,
                                    split="train",label_type="correct_choice",
                                    ds_method="pca",
                                    n_neighbors=20, min_dist=0.1,
                                    init="pca"):
    

    # 1. Get dimensionality reduction and plot
    fig, activations_ds_all = get_dim_reduction_and_plot(
        cfg,
        activations["activations_positive"],
        activations["activations_negative"],
        correct_choice = activations["correct_choice"],
        model_choice_positive=activations["model_choice_positive"],
        model_choice_negative=activations["model_choice_negative"],
        label_type=label_type,
        ds_method=ds_method,
        n_neighbors=n_neighbors, min_dist=min_dist, init=init
        )

    # 2. Save plot
    if ds_method == "pca":
        pca_path = os.path.join(cfg.artifact_path(), ds_method)
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(
            pca_path + os.sep + f"{cfg.model_name}_{ds_method}_{split}_{label_type}.html"
        )
        pio.write_image(
            fig,
            pca_path + os.sep + f"{cfg.model_name}_{ds_method}_{split}_{label_type}.pdf",
        )
        print(f"{ds_method} saved at {pca_path}")
    elif ds_method == "umap":
        pca_path = os.path.join(cfg.artifact_path(), 
                                ds_method,
                                f"init_{init}",
                                f"n_neighbors_{n_neighbors}",
                                f"min_dist_{min_dist}")
        if not os.path.exists(pca_path):
            os.makedirs(pca_path)
        fig.write_html(
            pca_path + os.sep + f"{cfg.model_name}_{ds_method}_{split}_{label_type}_{init}_{n_neighbors}_{min_dist}.html"
        )
        pio.write_image(
            fig,
            pca_path + os.sep + f"{cfg.model_name}_{ds_method}_{split}_{label_type}_{init}_{n_neighbors}_{min_dist}.pdf",
        )
        print(f"{ds_method} saved at {pca_path}")


def get_marker_symbol(choice):
    symbol_sequence = []
    for data in choice:
        if data == "A":
            symbol_sequence.append("star")
        elif data == "B":
            symbol_sequence.append("circle")
        elif data == "C":
            symbol_sequence.append("diamond")
        elif data == "D":
            symbol_sequence.append("triangle-down")
    symbol_sequence = np.array(symbol_sequence)

    return symbol_sequence


def get_dim_reduction(activations_all, n_layers, n_contrastive_data,
                       n_components=3, ds_method="pca",
                       n_neighbors=20, min_dist=0.1, init="pca"):
    activations_ds_all = np.zeros((n_layers, n_contrastive_data * 2, n_components))

    if ds_method == "pca":
        fit = PCA(n_components=n_components)
    elif ds_method == "umap":
        fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, init=init)

    for layer in range(n_layers):
            activations_ds = fit.fit_transform(
                    activations_all[layer, :, :].float().detach().cpu().numpy()
                )
            activations_ds_all[layer, :, :] = activations_ds

    return activations_ds_all


def get_dim_reduction_and_plot(cfg, activations_positive, activations_negative,
                     correct_choice=None,
                     model_choice_positive=None, model_choice_negative=None,
                     label_type="correct_choice",
                     ds_method="pca",
                     n_neighbors=20, min_dist=0.1, init="pca"):


    n_contrastive_data = activations_negative.shape[1]
    n_layers = activations_negative.shape[0]
    contrastive_type = cfg.contrastive_type

    print(f"n_contrastive_data: {n_contrastive_data}")
    print(f"n_layers: {n_layers}")
    print(f"contrastive_type: {contrastive_type}")

    # get symbole for the markers (either by model choice of correct choice)
    if label_type == "correct_choice":
        symbol_sequence_positive = get_marker_symbol(model_choice_positive)
        symbol_sequence_negative = get_marker_symbol(model_choice_negative)
    elif label_type == "model_choice":
        symbol_sequence_positive = get_marker_symbol(correct_choice)
        symbol_sequence_negative = get_marker_symbol(correct_choice)


    # 1. dimesionality reduction
    activations_all: Float[Tensor, "n_layers n_samples  d_model"] = torch.cat(
        (activations_positive, activations_negative), dim=1
    )
    activations_ds_all = get_dim_reduction(activations_all, n_layers, n_contrastive_data,
                                            n_components=3, ds_method=ds_method,
                                            n_neighbors=n_neighbors, min_dist=min_dist, init=init)
    
    
    # 2. plot
    cols = 7
    rows = math.ceil(n_layers / cols)
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=[f"layer {n}" for n in range(n_layers)]
    )      
    for row in range(rows):
        for ll, layer in enumerate(range(row * cols, row * cols + cols)):
            # print(f'layer{layer}')
            if layer < n_layers:

                
                df = {}
                df["pca0"] = activations_ds_all[layer, :, 0]
                df["pca1"] = activations_ds_all[layer, :, 1]
                df["pca2"] = activations_ds_all[layer, :, 2]

                fig.add_trace(
                    go.Scatter(
                        x=df["pca0"][:n_contrastive_data],
                        y=df["pca1"][:n_contrastive_data],
                        # z=df['pca2'][:n_contrastive_data],
                        mode="markers",
                        name=contrastive_type[0],
                        showlegend=False,
                        marker=dict(
                            symbol=symbol_sequence_positive,
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
                            symbol=symbol_sequence_negative,
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
                symbol="star",
                size=5,
                line=dict(width=1, color="darkslategrey"),
            ),
            name=f"{contrastive_type[0]}_A",
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
            name=f"{contrastive_type[1]}_A",
            marker_color="palevioletred",
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
            ),
            name=f"{contrastive_type[0]}_B",
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
                symbol="circle",
                size=5,
                line=dict(width=1, color="darkslategrey"),
            ),
            name=f"{contrastive_type[1]}_B",
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
    return fig, activations_ds_all