import os
import pickle
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_cos_sim_and_acc_layer(cfg):
    # 1. Load acc
    score_path = os.path.join(
        cfg.artifact_path(), "activation_steering", "positive_addition", "performance"
    )

    with open(
        score_path + os.sep + f"{cfg.model_alias}_honest_score_all_layers.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)

    score = data["score_negative_all"][:]  # last layer only

    # 2. Load cosine similarity (baseline)
    stage_path = os.path.join(cfg.artifact_path(), "stage_stats")
    with open(stage_path + os.sep + f"{cfg.model_alias}_stage_stats.pkl", "rb") as f:
        cos_data = pickle.load(f)
    cos = cos_data["stage_3"]["cos_positive_negative"]
    cos_pca = cos_data["stage_3"]["cos_positive_negative_pca"]

    # load cosine similarity (intervention)

    # 3. plot
    n_layers = len(cos)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=score,
            name="Performance",
            mode="lines+markers",
            marker=dict(color="dodgerblue"),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=cos,
            name="Cosine Similarity",
            mode="lines+markers",
            marker=dict(color="gold"),
            yaxis="y3",
        )
    )
    fig.update_xaxes(title_text="Layer")
    # fig.update_yaxes(title_text="Logit Difference", secondary_y=False)
    # fig.update_yaxes(title_text="Cosine Similarity", secondary_y=True)
    fig.update_layout(
        height=500,
        width=800,
        yaxis=dict(
            title="Performance",
            titlefont=dict(color="dodgerblue"),
            tickfont=dict(color="dodgerblue"),
        ),
        yaxis3=dict(
            title="Cosine Similarity",
            titlefont=dict(color="gold"),
            tickfont=dict(color="gold"),
            anchor="x",
            overlaying="y",
            side="right",
        ),
    )
    fig.show()
    fig.write_html(
        score_path + os.sep + f"{cfg.cache_name}_patch_cos_sim_layer" + ".html"
    )
    pio.write_image(
        fig,
        score_path + os.sep + f"{cfg.cache_name}_patch_cos_sim_layer" + ".png",
        scale=6,
    )
