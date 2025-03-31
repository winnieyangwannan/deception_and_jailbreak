from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import os
from MODEL.model_base import TransformerModelLoader
from PCA.activation_pca import get_contrastive_activation_dim_reduction
from CONFIGS.config_sandbag import Config
import torch
from EVALUATE.evaluate_sandbag import evaluate_generation
from DATA.dataset_sandbag import prepare_dataset
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch.utils.data.distributed import DistributedSampler
import random
from contrastive_steering_base import generate_and_steer_batch, save_activations
from STAGE_STATS.stage_statistics import get_state_quantification
import json


def generate_and_steer_batch_contrastive(cfg, model_base, dataset, accelerator,
                                         steering_vector=None, steering_method=None,
                                         do_pca=False, train_test="train",
                                         no_system=False):

    if accelerator.is_main_process:
        print(f"Preparing dataset")
    # 1. Prepare Dataset
    if no_system:
        dataloader_positive, dataset_positive = prepare_dataset(cfg, model_base, dataset, accelerator, 
                                                                contrastive_type=None)
        dataloader_negative, dataset_negative = prepare_dataset(cfg, model_base, dataset, accelerator, 
                                                                contrastive_type=None)
    else:
        dataloader_positive, dataset_positive = prepare_dataset(cfg, model_base, dataset, accelerator, 
                                                                contrastive_type=cfg.contrastive_type[0])
        dataloader_negative, dataset_negative = prepare_dataset(cfg, model_base, dataset, accelerator, 
                                                                contrastive_type=cfg.contrastive_type[1])
    
    dataloader_positive = accelerator.prepare(dataloader_positive)
    dataloader_negative = accelerator.prepare(dataloader_negative)
    
    # 2. Generate and steer for the entire dataset
    outputs_positive, cache_positive = generate_and_steer_batch(cfg, model_base, dataloader_positive, accelerator,
                                                steering_vector=steering_vector, steering_method=steering_method)
    outputs_negative, cache_negative = generate_and_steer_batch(cfg, model_base, dataloader_negative, accelerator,
                                                steering_vector=steering_vector, steering_method=steering_method)
    
    # 3. Save model outputs

    save_completions_two(cfg, outputs_positive, outputs_negative,
                      dataset_positive, dataset_negative, accelerator)
    
    # 4. Evaluate model outputs

    model_choice_positive_all, model_choice_negative_all = evaluate_generation(cfg, train_test=train_test)
    
    # 5. Save cache and do PCA

    if do_pca:
        if accelerator.is_main_process:

            activations = save_activations(cfg, cache_positive, cache_negative,
                                           dataset["correct_choice"],
                                           model_choice_positive_all, model_choice_negative_all)
    
            # Get the PCA of the activations and visualize it
            get_contrastive_activation_dim_reduction(cfg, activations, split=train_test,
            label_type="correct_choice")
            get_contrastive_activation_dim_reduction(cfg, activations, split=train_test,
            label_type="model_choice")
    if do_pca:
        if accelerator.is_main_process:
            get_state_quantification(
                    cfg,
                    activations["activations_positive"],
                    activations["activations_negative"],
                    activations["correct_choice"],
                    save_plot=True,
                    save_name=train_test,
                )



def save_completions(cfg, output_positive, output_negative, 
                     dataset_positive, dataset_negative, accelerator, train_test="train" ):
    completions = {
        f"prompt_{cfg.contrastive_type[0]}": dataset_positive["messages"][0],
        f"prompt_{cfg.contrastive_type[1]}": dataset_negative["messages"][0]}

    completions["completions"] = [
        {
            "question": dataset_positive[f"question"][i],
            "answer": dataset_positive[f"answer"][i],
            "correct_choice": dataset_positive[f"correct_choice"][i],
            "A": dataset_positive[f"A"][i],
            "B": dataset_positive[f"B"][i],
            "C": dataset_positive[f"C"][i],
            "D": dataset_positive[f"D"][i],
            f"output_{cfg.contrastive_type[0]}": output_positive[i],
            f"output_{cfg.contrastive_type[1]}": output_negative[i],
            "ID": i,
        }
        for i in range(len(output_positive))
    ]


    completion_path = os.path.join(cfg.artifact_path(), "completions")
    if not os.path.exists(completion_path):
        os.makedirs(completion_path)
    with open(
        completion_path + os.sep + f"{cfg.model_name}_completions_{train_test}.json",
        "w",
    ) as f:
        json.dump(completions, f, indent=4)
    if accelerator.is_main_process:
        print(f"completions saved at {completion_path}")



def save_completions_two(cfg, output_positive, output_negative, 
                         dataset_positive, dataset_negative, accelerator, train_test="train" ):
    completions = {
        f"prompt_{cfg.contrastive_type[0]}": dataset_positive["messages"][0],
        f"prompt_{cfg.contrastive_type[1]}": dataset_negative["messages"][0]}
    completions["completions"] = [
        {
            "question": dataset_positive[f"question"][i],
            "correct_answer": dataset_positive[f"correct_answer"][i],
            "wrong_answer": dataset_positive[f"wrong_answer"][i],
            "correct_choice": dataset_positive[f"correct_choice"][i],
            f"output_{cfg.contrastive_type[0]}": output_positive[i],
            f"output_{cfg.contrastive_type[1]}": output_negative[i],
            "ID": i,
        }
        for i in range(len(output_positive))
    ]


    completion_path = os.path.join(cfg.artifact_path(), "completions")
    if not os.path.exists(completion_path):
        os.makedirs(completion_path)
    with open(
        completion_path + os.sep + f"{cfg.model_name}_completions_{train_test}.json",
        "w",
    ) as f:
        json.dump(completions, f, indent=4)
    if accelerator.is_main_process:
        print(f"completions saved at {completion_path}")