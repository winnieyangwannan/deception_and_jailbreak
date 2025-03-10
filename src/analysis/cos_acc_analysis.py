import os
import pickle
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


def collect_acc(cfg, checkpoint):
    if "Llama" in cfg.model_alias:
        with open(
            os.path.join(
                cfg.artifact_path(),
                str(checkpoint),
                "completions",
                f"{cfg.model_alias}_{cfg.task_name}_lora_True_completions_test.json",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            completion = json.load(f)
    else:
        with open(
            os.path.join(
                cfg.artifact_path(),
                str(checkpoint),
                "completions",
                f"{cfg.model_alias}_honest_lying_{cfg.task_name}_lora_True_completions_test.json",
            ),
            "r",
            encoding="utf-8",
        ) as f:
            completion = json.load(f)

    honest_score_honest = completion["honest_score_honest"]
    honest_score_lying = completion["honest_score_lying"]
    honest_score_honest_before = completion["honest_score_honest_before"]
    honest_score_lying_before = completion["honest_score_lying_before"]

    return (
        honest_score_honest,
        honest_score_lying,
        honest_score_honest_before,
        honest_score_lying_before,
    )


def collect_cos(cfg, checkpoint):
    if "Llama" in cfg.model_alias:
        with open(
            os.path.join(
                cfg.artifact_path(),
                str(checkpoint),
                "stage_stats",
                f"{cfg.model_alias}_{cfg.task_name}_lora_True_stage_stats_test.pkl",
            ),
            "rb",
        ) as f:
            stage_stats = pickle.load(f)
        cos = stage_stats["stage_3"]["cos_positive_negative"]  # cos in original high d
    else:
        with open(
            os.path.join(
                cfg.artifact_path(),
                str(checkpoint),
                "stage_stats",
                f"{cfg.model_alias}_honest_lying_{cfg.task_name}_lora_True_stage_stats_test.pkl",
            ),
            "rb",
        ) as f:
            stage_stats = pickle.load(f)
        cos = stage_stats["stage_3"]["cos_positive_negative"]  # cos in original high d
    return cos


def collect_all_cos_acc(cfg):

    cos_all = []
    honest_score_honest_all = []
    honest_score_lying_all = []
    honest_score_honest_before_all = []
    honest_score_lying_before_all = []
    # 1.collect cosine similarity for each checkpoint

    for checkpoint in cfg.checkpoints[1:]:

        (
            honest_score_honest,
            honest_score_lying,
            honest_score_honest_before,
            honest_score_lying_before,
        ) = collect_acc(cfg, checkpoint)
        honest_score_honest_all.append(honest_score_honest)
        honest_score_lying_all.append(honest_score_lying)
        honest_score_honest_before_all.append(honest_score_honest_before)
        honest_score_lying_before_all.append(honest_score_lying_before)

        # take the average of all scores before manipulation (sft, icl etc)
        honest_score_honest_before = np.mean(honest_score_honest_before_all)
        honest_score_lying_before = np.mean(honest_score_lying_before_all)

        # collect performance (honest_score) for each checkpoint
        cos = collect_cos(cfg, checkpoint)
        # take the inversion of cosine --> rotation progress
        cos_all.append(-cos[-1])  # only take last layer and

    # take the starting point as the honest score before manipulation
    honest_score_honest_all.insert(0, honest_score_honest_before)
    honest_score_lying_all.insert(0, honest_score_lying_before)

    # 2. take the cosine before manipulation
    with open(
        os.path.join(
            cfg.data_dir,
            "runs",
            "activation_pca",
            cfg.model_alias,
            "stage_stats",
            f"{cfg.model_alias}_stage_stats.pkl",
        ),
        "rb",
    ) as f:
        stage_stats = pickle.load(f)
    cos_0 = stage_stats["stage_3"]["cos_positive_negative"]  # cos in original high d
    cos_all.insert(0, -cos_0[-1])
    cos_all_norm = (cos_all - np.min(cos_all)) / (np.max(cos_all) - np.min(cos_all))

    return cos_all_norm, honest_score_honest_all, honest_score_lying_all


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

    # 2. Load cosine similarity
    stage_path = os.path.join(
        cfg.artifact_path(), "activation_steering", "positive_addition", "stage_stats"
    )
    with open(
        stage_path + os.sep + f"{cfg.model_alias}_cos_sim_all_layers.pkl", "rb"
    ) as f:
        cos_data = pickle.load(f)
    cos = [cos_data["cos_all"][ii][-1] * -11 for ii in range(len(cos_data["cos_all"]))]
    cos_all_norm = (cos - np.min(cos)) / (np.max(cos) - np.min(cos))

    # load cosine similarity (intervention)

    # 3. plot
    n_layers = len(cos_all_norm)
    line_width = 5
    size = 12
    font_size = 20
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=score,
            name="Performance",
            mode="lines+markers",
            marker=dict(color="dodgerblue", size=size),
            line=dict(color="dodgerblue", width=line_width),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(n_layers),
            y=cos_all_norm,
            name="Rotation Progress",
            mode="lines+markers",
            marker=dict(color="gold", size=size),
            yaxis="y3",
            line=dict(color="gold", width=line_width),
        )
    )
    fig.update_xaxes(title_text="Layer")
    # fig.update_yaxes(title_text="Logit Difference", secondary_y=False)
    # fig.update_yaxes(title_text="Cosine Similarity", secondary_y=True)

    fig.update_layout(
        height=400,
        width=600,
        yaxis=dict(
            title="Performance",
            titlefont=dict(color="dodgerblue"),
            tickfont=dict(color="dodgerblue"),
            tickmode="array",
            tickvals=[0, 0.5, 1],
            range=[-0.01, 1.11],
        ),
        yaxis3=dict(
            title="Rotation Progress",
            titlefont=dict(color="gold"),
            tickfont=dict(color="gold"),
            anchor="x",
            overlaying="y",
            side="right",
            tickmode="array",
            tickvals=[0, 0.5, 1],
            range=[-0.01, 1.11],
            # ticktext=["0", ""],
        ),
        font=dict(size=font_size),
    )

    fig.show()

    pio.write_image(
        fig,
        cfg.output_path()
        + os.sep
        + f"{cfg.model_alias}_steer_to_honest_acc_cos_layer"
        + ".pdf",
    )


def plot_cos_sim_and_acc_checkpoint(cfg, cos_all, honest_score_all):

    # plot
    n_checkpoints = len(cos_all)
    n_samples = [cp * cfg.batch_size for cp in cfg.checkpoints]
    line_width = 5
    size = 12
    font_size = 20
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=n_samples,
            y=honest_score_all,
            name="Performance",
            mode="lines+markers",
            marker=dict(color="dodgerblue", size=size),
            line=dict(color="dodgerblue", width=line_width),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=n_samples,
            y=cos_all,
            name="Rotation Progress",
            mode="lines+markers",
            marker=dict(color="gold", size=size),
            yaxis="y3",
            line=dict(color="gold", width=line_width),  # ,dash='dash'),
        )
    )
    fig.update_xaxes(title_text="# of Training Examples")

    fig.update_layout(
        height=400,
        width=600,
        showlegend=True,
        xaxis=dict(
            tickmode="array",
        ),
        yaxis=dict(
            title="Performance",
            titlefont=dict(color="dodgerblue"),
            tickfont=dict(color="dodgerblue"),
            tickmode="array",
            tickvals=[0, 0.5, 1],
            range=[-0.01, 1.11],
        ),
        yaxis3=dict(
            title="Rotation Progress",
            titlefont=dict(color="gold"),
            tickfont=dict(color="gold"),
            anchor="x",
            overlaying="y",
            side="right",
            tickmode="array",
            tickvals=[0, 0.5, 1],
            range=[-0.01, 1.11],
        ),
        font=dict(size=font_size),
    )

    fig.show()

    pio.write_image(
        fig,
        cfg.save_path()
        + os.sep
        + f"{cfg.model_alias}_{cfg.task_name}_acc_cos_checkpoints"
        + ".pdf",
    )
