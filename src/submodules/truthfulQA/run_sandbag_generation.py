from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import os
from MODEL.model_base import TransformerModelLoader
from contrastive_steering_sandbag import generate_and_steer_batch_contrastive
from CONFIGS.config_sandbag import Config
import torch
from EVALUATE.evaluate_sandbag import evaluate_generation_train
from DATA.dataset_sandbag import  preprocess_dataset, preprocess_dataset_two
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch.utils.data.distributed import DistributedSampler
import random
from contrastive_steering_base import generate_and_steer_batch, save_activations
from STAGE_STATS.stage_statistics import get_state_quantification



# get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
os.chdir(current_dir)
current_directory = os.getcwd()
print("current_directory:", current_directory)


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_path", 
        type=str,
        required=False, 
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Path to the base model"
    )
    parser.add_argument(
        "--category", 
        type=str,
        required=False, 
        default="wmdp-bio",
        help="category of the dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default="/home/winnieyangwn/Output",
        help="Path to save the results",
        # ~/local/steering/output for dev server
    )
    parser.add_argument(
        "--source_layer",
        type=int,
        required=False,
        default=14,
        help="layer to extract the steering vector",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        required=False,
        default=14,
        help="layer to steer the model",
    )
    parser.add_argument(
        "--steering_strength",
        type=int,
        required=False,
        default=1,
        help="coefficient to multiply the steering vector",
    )
    parser.add_argument(
        "--system_type",
        type=str,
        required=False,
        default="sandbag", #misconception
        help="",
    )
    return parser.parse_args()


def main(model_path, save_dir,
         source_layer=14, target_layer=14, steering_strength=1,
         category="wmdp-chem",
         system_type="misconception",): #misconception):
    
    # Initialize accelerator
    accelerator = Accelerator()
    # Set seed for reproducibility across processes
    set_seed(42)
    if system_type == "misconception":
        contrastive_type =  ("factual", "misconception")
    elif system_type == "lie":
        contrastive_type =  ("honest", "lying")
    elif system_type == "sandbag":
        contrastive_type =  ("control", "sandbag")
    # 0. cfg
    cfg = {}
    model_name = os.path.basename(model_path)
    # Create config
    cfg = Config(model_path=model_path,     
                model_name=model_name,
                category=category,
                save_dir=save_dir,
                source_layer=source_layer, target_layer=target_layer,
                steering_strength=steering_strength,
                system_type=system_type,
                contrastive_type=contrastive_type,)

    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Process rank: {accelerator.process_index}")
        print(f"cfg: {cfg}")

    # 1. Load dataset and train-test split
    # dataset = load_dataset("csv", data_files="DATA/TruthfulQA.csv")["train"]
    dataset = load_dataset("cais/wmdp", cfg.category)["test"]


    # train test split
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    dataset_train = split_dataset["train"]
    dataset_test = split_dataset["test"]

    if accelerator.is_main_process:
        print(f"Train size: {len(dataset_train)}")
        print(f"Test size: {len(dataset_test)}")
    
    if cfg.n_train > 0 and cfg.n_train < len(dataset_train):
        dataset_train =  dataset_train.select(range(cfg.n_train)) 
        dataset_test =  dataset_test.select(range(cfg.n_test))
    if cfg.batch_size > len(dataset_train) or cfg.batch_size == 0:
        cfg.batch_size = len(dataset_train)


    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Train size (SUBSAMPLE): {len(dataset_train)}")
        print(f"Test size (SUBSAMPLE): {len(dataset_test)}")


    dataset_train = preprocess_dataset_two(dataset_train)

    
    # 2. Load model
    model_base = TransformerModelLoader(cfg, accelerator=accelerator)

    # 3. Generate Training Examples
    generate_and_steer_batch_contrastive(cfg, model_base, dataset_train, accelerator,
                                         )

    print("done")


if __name__ == "__main__":
    args = parse_arguments()
    main(model_path=args.model_path,
         save_dir=args.save_dir,
         source_layer=args.source_layer, 
         target_layer=args.target_layer,
         steering_strength=args.steering_strength,
         category=args.category,
         system_type=args.system_type)

