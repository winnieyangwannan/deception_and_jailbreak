from pipeline.configs.config_generation import Config

import json
import os
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pprint
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from sklearn.metrics import confusion_matrix
from src.submodules.evaluations.evaluate_deception import (
    get_honest_score,
    get_lying_score,
    get_unexpected_rate,
)


def evaluate_generation_deception_SFT_chinese(
    cfg,
    save_path=None,
    save_name=None,
    train_test="train",
):
    """
    Evaluate generation result for the deception project.
    :param cfg:
    :param contrastive_type:
    :param save_path:
    :param save_name:
    :return:
    """
    if save_path is None:
        save_path = os.path.join(cfg.artifact_path(), "completions")

    # (1) Load the completion result
    if save_name is None:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{train_test}.json",
            "r",
        ) as f:
            completions = json.load(f)
    else:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{save_name}_{train_test}.json",
            "r",
        ) as f:
            completions = json.load(f)

    if type(completions) == dict:
        completions = completions["completions"]

    # (2) For each response, get the honest score, lying score, and unexpected rate
    for completion in completions:
        completion["response_honest"] = int(
            get_honest_score(completion["response_honest"], int(completion["label"]))
        )
        completion["response_lying"] = int(
            get_honest_score(completion["response_lying"], int(completion["label"]))
        )

        ######################################################################
        completion["response_honest"] = int(
            get_lying_score(completion["response_honest"], int(completion["label"]))
        )
        completion["response_lying"] = int(
            get_lying_score(completion["response_lying"], int(completion["label"]))
        )

        ######################################################################
        unexpected, predicted_label = get_unexpected_rate(
            completion["response_honest"], int(completion["label"])
        )
        completion["unexpected_honest"] = int(unexpected)
        completion["predicted_honest"] = predicted_label

        unexpected, predicted_label = get_unexpected_rate(
            completion["response_lying"], int(completion["label"])
        )
        completion["unexpected_lying"] = int(unexpected)
        completion["predicted_label_lying"] = predicted_label

    # (3) Per category summary statistics
    # category_to_asr = {}
    # categories = ["false", "true"]
    # for category in categories:
    #     category_completions = [
    #         completion for completion in completions if completion["label"] == category
    #     ]
    #     category_honest_score = np.mean(
    #         [completion["honest_score"] for completion in category_completions]
    #     )
    #     category_lying_score = np.mean(
    #         [completion["lying_score"] for completion in category_completions]
    #     )
    #     category_unexpected = np.mean(
    #         [completion["unexpected"] for completion in category_completions]
    #     )
    #     category_to_asr[f"honest_score_label_{category}"] = category_honest_score
    #     category_to_asr[f"lying_score_label_{category}"] = category_lying_score
    #     category_to_asr[f"unexpected_label_{category}"] = category_unexpected

    # (4) Mean summary statistics
    evaluation = {}
    # mean over all completions
    evaluation["honest_score_honest"] = np.mean(
        [completion["honest_score_honest"] for completion in completions]
    )
    evaluation["honest_score_lying"] = np.mean(
        [completion["honest_score_lying"] for completion in completions]
    )

    evaluation["lying_score_honest"] = np.mean(
        [completion["lying_score_honest"] for completion in completions]
    )
    evaluation["lying_score_lying"] = np.mean(
        [completion["lying_score_lying"] for completion in completions]
    )

    evaluation["unexpected_honest"] = np.mean(
        [completion["unexpected_honest"] for completion in completions]
    )
    evaluation["unexpected_lying"] = np.mean(
        [completion["unexpected_lying"] for completion in completions]
    )

    # evaluation["score_per_category"] = category_to_asr
    completion_evaluation = evaluation
    completion_evaluation["completions"] = completions

    # print(evaluation)

    # (5) Save completion results
    if save_name is None:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{train_test}.json",
            "w",
        ) as f:
            json.dump(completion_evaluation, f, indent=4)
            print(f"Evaluation results saved at {save_path}")
    else:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{save_name}_{train_test}.json",
            "w",
        ) as f:
            json.dump(completion_evaluation, f, indent=4)
            print(f"Evaluation results saved at {save_path}")

    return completion_evaluation


def evaluate_generation_deception_SFT(
    cfg,
    save_path=None,
    save_name=None,
    train_test="train",
):
    """
    Evaluate generation result for the deception project.
    :param cfg:
    :param contrastive_type:
    :param save_path:
    :param save_name:
    :return:
    """
    if save_path is None:
        save_path = os.path.join(cfg.artifact_path(), "completions")

    # (1) Load the completion result
    if save_name is None:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{train_test}.json",
            "r",
        ) as f:
            completions = json.load(f)
    else:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{save_name}_{train_test}.json",
            "r",
        ) as f:
            completions = json.load(f)

    if type(completions) == dict:
        completions = completions["completions"]

    # (2) For each response, get the honest score, lying score, and unexpected rate
    for completion in completions:
        completion["honest_score_big_honest"] = int(
            get_honest_score(
                completion["response_big_honest"], int(completion["label"])
            )
        )
        completion["honest_score_big_lying"] = int(
            get_honest_score(completion["response_big_lying"], int(completion["label"]))
        )
        completion["honest_score_small_before_honest"] = int(
            get_honest_score(
                completion["response_small_before_honest"], int(completion["label"])
            )
        )
        completion["honest_score_small_before_lying"] = int(
            get_honest_score(
                completion["response_small_before_lying"], int(completion["label"])
            )
        )
        completion["honest_score_small_after_honest"] = int(
            get_honest_score(
                completion["response_small_after_honest"], int(completion["label"])
            )
        )
        completion["honest_score_small_after_lying"] = int(
            get_honest_score(
                completion["response_small_after_lying"], int(completion["label"])
            )
        )

        ######################################################################
        completion["lying_score_big_honest"] = int(
            get_lying_score(completion["response_big_honest"], int(completion["label"]))
        )
        completion["lying_score_big_lying"] = int(
            get_lying_score(completion["response_big_lying"], int(completion["label"]))
        )
        completion["lying_score_small_before_honest"] = int(
            get_lying_score(
                completion["response_small_before_honest"], int(completion["label"])
            )
        )
        completion["lying_score_small_before_lying"] = int(
            get_lying_score(
                completion["response_small_before_lying"], int(completion["label"])
            )
        )
        completion["lying_score_small_after_honest"] = int(
            get_lying_score(
                completion["response_small_after_honest"], int(completion["label"])
            )
        )
        completion["lying_score_small_after_lying"] = int(
            get_lying_score(
                completion["response_small_after_lying"], int(completion["label"])
            )
        )

        ######################################################################
        unexpected, predicted_label = get_unexpected_rate(
            completion["response_big_honest"], int(completion["label"])
        )
        completion["unexpected_big_honest"] = int(unexpected)
        completion["predicted_label_big_honest"] = predicted_label

        unexpected, predicted_label = get_unexpected_rate(
            completion["response_big_lying"], int(completion["label"])
        )
        completion["unexpected_big_lying"] = int(unexpected)
        completion["predicted_label_big_lying"] = predicted_label

        #
        unexpected, predicted_label = get_unexpected_rate(
            completion["response_small_before_honest"], int(completion["label"])
        )
        completion["unexpected_small_before_honest"] = int(unexpected)
        completion["predicted_label_small_before_honest"] = predicted_label

        unexpected, predicted_label = get_unexpected_rate(
            completion["response_small_before_lying"], completion["label"]
        )
        completion["unexpected_small_before_lying"] = int(unexpected)
        completion["predicted_label_small_before_lying"] = predicted_label

        #
        unexpected, predicted_label = get_unexpected_rate(
            completion["response_small_after_honest"], int(completion["label"])
        )
        completion["unexpected_small_after_honest"] = int(unexpected)
        completion["predicted_label_small_after_honest"] = predicted_label

        unexpected, predicted_label = get_unexpected_rate(
            completion["response_small_after_lying"], int(completion["label"])
        )
        completion["unexpected_small_after_lying"] = int(unexpected)
        completion["predicted_label_small_after_lying"] = predicted_label

    # (3) Per category summary statistics
    # category_to_asr = {}
    # categories = ["false", "true"]
    # for category in categories:
    #     category_completions = [
    #         completion for completion in completions if completion["label"] == category
    #     ]
    #     category_honest_score = np.mean(
    #         [completion["honest_score"] for completion in category_completions]
    #     )
    #     category_lying_score = np.mean(
    #         [completion["lying_score"] for completion in category_completions]
    #     )
    #     category_unexpected = np.mean(
    #         [completion["unexpected"] for completion in category_completions]
    #     )
    #     category_to_asr[f"honest_score_label_{category}"] = category_honest_score
    #     category_to_asr[f"lying_score_label_{category}"] = category_lying_score
    #     category_to_asr[f"unexpected_label_{category}"] = category_unexpected

    # (4) Mean summary statistics
    evaluation = {}
    # mean over all completions
    evaluation["honest_score_big_honest"] = np.mean(
        [completion["honest_score_big_honest"] for completion in completions]
    )
    evaluation["honest_score_big_lying"] = np.mean(
        [completion["honest_score_big_lying"] for completion in completions]
    )

    evaluation["honest_score_small_before_honest"] = np.mean(
        [completion["honest_score_small_before_honest"] for completion in completions]
    )
    evaluation["honest_score_small_before_lying"] = np.mean(
        [completion["honest_score_small_before_lying"] for completion in completions]
    )

    evaluation["honest_score_small_after_honest"] = np.mean(
        [completion["honest_score_small_after_honest"] for completion in completions]
    )
    evaluation["honest_score_small_after_lying"] = np.mean(
        [completion["honest_score_small_after_lying"] for completion in completions]
    )

    evaluation["lying_score_big_honest"] = np.mean(
        [completion["lying_score_big_honest"] for completion in completions]
    )
    evaluation["lying_score_big_lying"] = np.mean(
        [completion["lying_score_big_lying"] for completion in completions]
    )
    evaluation["lying_score_small_before_honest"] = np.mean(
        [completion["lying_score_small_before_honest"] for completion in completions]
    )
    evaluation["lying_score_small_before_lying"] = np.mean(
        [completion["lying_score_small_before_lying"] for completion in completions]
    )
    evaluation["lying_score_small_after_honest"] = np.mean(
        [completion["lying_score_small_after_honest"] for completion in completions]
    )
    evaluation["lying_score_small_after_lying"] = np.mean(
        [completion["lying_score_small_after_lying"] for completion in completions]
    )

    evaluation["unexpected_big_honest"] = np.mean(
        [completion["unexpected_big_honest"] for completion in completions]
    )
    evaluation["unexpected_big_lying"] = np.mean(
        [completion["unexpected_big_lying"] for completion in completions]
    )
    evaluation["unexpected_small_before_honest"] = np.mean(
        [completion["unexpected_small_before_honest"] for completion in completions]
    )
    evaluation["unexpected_small_before_lying"] = np.mean(
        [completion["unexpected_small_before_lying"] for completion in completions]
    )
    evaluation["unexpected_small_after_honest"] = np.mean(
        [completion["unexpected_small_after_honest"] for completion in completions]
    )
    evaluation["unexpected_small_after_lying"] = np.mean(
        [completion["unexpected_small_after_lying"] for completion in completions]
    )
    # evaluation["score_per_category"] = category_to_asr
    completion_evaluation = evaluation
    completion_evaluation["completions"] = completions

    # print(evaluation)

    # (5) Save completion results
    if save_name is None:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{train_test}.json",
            "w",
        ) as f:
            json.dump(completion_evaluation, f, indent=4)
            print(f"Evaluation results saved at {save_path}")
    else:
        with open(
            save_path
            + os.sep
            + f"{cfg.model_name_small}-{cfg.model_name_big}_completions_{save_name}_{train_test}.json",
            "w",
        ) as f:
            json.dump(completion_evaluation, f, indent=4)
            print(f"Evaluation results saved at {save_path}")

    return completion_evaluation


def plot_performance_bar(evaluation, save_path, save_name=None, train_test="train"):

    honest_score_before = [
        evaluation["honest_score_small_before_honest"],
        evaluation["honest_score_small_before_lying"],
    ]
    honest_score_after = [
        evaluation["honest_score_small_after_honest"],
        evaluation["honest_score_small_after_lying"],
    ]

    lying_score_before = [
        evaluation["lying_score_small_before_honest"],
        evaluation["lying_score_small_before_lying"],
    ]
    lying_score_after = [
        evaluation["lying_score_small_after_honest"],
        evaluation["lying_score_small_after_lying"],
    ]

    unexpected_before = [
        evaluation["unexpected_small_before_honest"],
        evaluation["unexpected_small_before_lying"],
    ]
    unexpected_after = [
        evaluation["unexpected_small_after_honest"],
        evaluation["unexpected_small_after_lying"],
    ]
    role = ["Honest", "Lying"]

    # plot
    fig = make_subplots(
        rows=1,
        cols=6,
        subplot_titles=[
            "Honesty Score Before",
            "Honesty Score After",
            "Lying Score Before",
            "Lying Score After",
            "Unexpected Before",
            "Unexpected After",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=role,
            y=honest_score_before,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=role,
            y=honest_score_after,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=role,
            y=lying_score_before,
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=role,
            y=lying_score_after,
            showlegend=False,
        ),
        row=1,
        col=4,
    )

    fig.add_trace(
        go.Bar(
            x=role,
            y=unexpected_before,
            showlegend=False,
        ),
        row=1,
        col=5,
    )
    fig.add_trace(
        go.Bar(
            x=role,
            y=unexpected_after,
            showlegend=False,
        ),
        row=1,
        col=6,
    )

    fig["layout"]["yaxis"]["title"] = "Frequency"
    fig["layout"]["xaxis"]["title"] = "Persona"

    fig.update_layout(yaxis1=dict(range=[0, 1]))
    fig.update_layout(yaxis2=dict(range=[0, 1]))
    fig.update_layout(yaxis3=dict(range=[0, 1]))
    fig.update_layout(yaxis4=dict(range=[0, 1]))
    fig.update_layout(yaxis5=dict(range=[0, 1]))
    fig.update_layout(yaxis6=dict(range=[0, 1]))

    fig.show()
    if save_name is not None:
        pio.write_image(
            fig,
            save_path + os.sep + f"performance_{save_name}_{train_test}.png",
            scale=6,
        )
    else:
        pio.write_image(
            fig, save_path + os.sep + f"performance_{train_test}.png", scale=6
        )
    return fig


def plot_performance_bar_chinese(
    evaluation, save_path, save_name=None, train_test="train"
):

    honest_score = [
        evaluation["honest_score_honest"],
        evaluation["honest_score_lying"],
    ]

    lying_score = [
        evaluation["lying_score_honest"],
        evaluation["lying_score_lying"],
    ]

    unexpected = [
        evaluation["unexpected_honest"],
        evaluation["unexpected_lying"],
    ]

    role = ["Honest", "Lying"]

    # plot
    fig = make_subplots(
        rows=1,
        cols=6,
        subplot_titles=[
            "Honesty Score Before",
            "Honesty Score After",
            "Lying Score Before",
            "Lying Score After",
            "Unexpected Before",
            "Unexpected After",
        ],
    )
    fig.add_trace(
        go.Bar(
            x=role,
            y=honest_score,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=role,
            y=lying_score,
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Bar(
            x=role,
            y=unexpected,
            showlegend=False,
        ),
        row=1,
        col=5,
    )

    fig["layout"]["yaxis"]["title"] = "Frequency"
    fig["layout"]["xaxis"]["title"] = "Persona"

    fig.update_layout(yaxis1=dict(range=[0, 1]))
    fig.update_layout(yaxis2=dict(range=[0, 1]))
    fig.update_layout(yaxis3=dict(range=[0, 1]))
    fig.update_layout(yaxis4=dict(range=[0, 1]))
    fig.update_layout(yaxis5=dict(range=[0, 1]))
    fig.update_layout(yaxis6=dict(range=[0, 1]))

    fig.show()
    if save_name is not None:
        pio.write_image(
            fig,
            save_path + os.sep + f"performance_{save_name}_{train_test}.png",
            scale=6,
        )
    else:
        pio.write_image(
            fig, save_path + os.sep + f"performance_{train_test}.png", scale=6
        )
    return fig


def plot_performance_confusion_matrix(
    true_label,
    predicted_label,
    contrastive_type=None,
    save_path=None,
    save_name=None,
    save_fig=True,
):
    for ii, label in enumerate(true_label):
        if label == 0:
            true_label[ii] = "false"
        elif label == 1:
            true_label[ii] = "true"

    for ii, label in enumerate(predicted_label):
        if label == -1:
            predicted_label[ii] = "unexpected"

    cm = confusion_matrix(true_label, predicted_label)
    cmn = confusion_matrix(true_label, predicted_label, normalize="true")

    for ii in range(cmn.shape[0]):
        for jj in range(cmn.shape[1]):
            cmn[ii, jj] = round(cmn[ii, jj], 2)

    fig = px.imshow(
        cmn,
        text_auto=True,
        width=500,
        height=500,
    )
    # fig = go.Figure(
    #     data=go.Heatmap(
    #         z=cm,
    #         text=cm,
    #         texttemplate="%{text}",
    #     ),
    # )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["False", "True"]),
        yaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["False", "True"]),
    )

    if save_fig:

        fig.show()
        if save_name is not None:

            pio.write_image(
                fig,
                save_path
                + os.sep
                + f"confusion_matrix_{save_name}_{contrastive_type}.png",
                scale=6,
            )
        else:
            pio.write_image(
                fig,
                save_path + os.sep + "confusion_matrix_" + contrastive_type + ".png",
                scale=6,
            )

    return fig


def plot_lying_honest_performance(
    cfg, completions, save_path=None, save_name=None, train_test="test"
):
    # contrastive_type = cfg.contrastive_type
    if save_path is None:
        save_path = os.path.join(cfg.artifact_path(), "performance")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # true_label_honest = [
    #     completion["label"] for completion in completions_honest["completions"]
    # ]
    #
    # predicted_label_honest = [
    #     completion["predicted_label"]
    #     for completion in completions_honest["completions"]
    # ]
    # predicted_label_lying = [
    #     completion["predicted_label"] for completion in completions_lying["completions"]
    # ]
    if cfg.task_name == "SFT":
        plot_performance_bar(completions, save_path, save_name, train_test)
    elif cfg.task_name == "SFT_chinese":
        plot_performance_bar_chinese(completions, save_path, save_name, train_test)
    # plot_performance_confusion_matrix(
    #     true_label_honest,
    #     predicted_label_honest,
    #     contrastive_type[0],
    #     save_path,
    #     save_name,
    # )
    # plot_performance_confusion_matrix(
    #     true_label_honest,
    #     predicted_label_lying,
    #     contrastive_type[1],
    #     save_path,
    #     save_name,
    # )
