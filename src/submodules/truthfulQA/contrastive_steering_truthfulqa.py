from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import os
from MODEL.model_base import TransformerModelLoader
from PCA.activation_pca import get_contrastive_activation_dim_reduction
from CONFIGS.config_truthfulqa import Config
import torch
from EVALUATE.evaluate_truthful_qa import evaluate_generation
from DATA.dataset_truthfulqa import prepare_dataset
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
                                         train_test="train",
                                         do_pca=False, no_system=False):
    
        
    # 1. Prepare Dataset
        
    if accelerator.is_main_process:
        print(f"Preparing dataset")
    if no_system:
        dataloader_positive, dataset_positive = prepare_dataset(cfg, model_base,
                                                                dataset, accelerator, 
                                                                contrastive_type=None)
        dataloader_negative, dataset_negative = prepare_dataset(cfg, model_base, 
                                                                dataset, accelerator, 
                                                                contrastive_type=None)
    
    else:
        dataloader_positive, dataset_positive = prepare_dataset(cfg, model_base,
                                                                dataset, accelerator, 
                                                                contrastive_type=cfg.contrastive_type[0])
        dataloader_negative, dataset_negative = prepare_dataset(cfg, model_base, 
                                                                dataset, accelerator, 
                                                                contrastive_type=cfg.contrastive_type[1])
        
    if accelerator.is_main_process:
        print(f"Preparing dataloader")
    dataloader_positive = accelerator.prepare(dataloader_positive)
    dataloader_negative = accelerator.prepare(dataloader_negative)
    
    # 2. Generate and steer for the entire dataset
    if accelerator.is_main_process:
        print(f"generate and steer:")

    outputs_positive, cache_positive = generate_and_steer_batch(cfg, model_base, dataloader_positive,
                                                                 accelerator,
                                                                 steering_vector=steering_vector, 
                                                                 steering_method=steering_method)
    outputs_negative, cache_negative = generate_and_steer_batch(cfg, model_base, dataloader_negative,
                                                                 accelerator,
                                                                 steering_vector=steering_vector,
                                                                 steering_method=steering_method)
    
    # 3. Save model outputs

    save_completions(cfg, outputs_positive, outputs_negative,
                      dataset_positive, dataset_negative, accelerator)
    
    # 4. Evaluate model outputs

    model_choice_positive_all, model_choice_negative_all = evaluate_generation(cfg, train_test=train_test)
    
    # 5. Save activation cache, do PCA and stage quantification

    if do_pca:
        activations = save_activations(cfg, cache_positive, cache_negative,
                                        dataset["Correct choice"],
                                        model_choice_positive_all, model_choice_negative_all)

        # Get the PCA of the activations and visualize it
        get_contrastive_activation_dim_reduction(cfg, activations, split=train_test,
        label_type="correct_choice")
        get_contrastive_activation_dim_reduction(cfg, activations, split=train_test,
        label_type="model_choice")

        # # Get the stage quantification of the activations
        if accelerator.is_main_process:
            print(f"get_state_quantification")
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
            "Category": dataset_positive[f"Category"][i],
            "Question": dataset_positive[f"Question"][i],
            "Correct choice": dataset_positive[f"Correct choice"][i],
            "Best Answer": dataset_positive[f"Best Answer"][i],
            "Best Incorrect Answer": dataset_positive[f"Best Incorrect Answer"][i],
            "Source": dataset_positive[f"Source"][i],
            f"Output_{cfg.contrastive_type[0]}": output_positive[i],
            f"Output_{cfg.contrastive_type[1]}": output_negative[i],
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


