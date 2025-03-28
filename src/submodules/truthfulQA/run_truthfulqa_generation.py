from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import os
from model_base import TransformerModelLoader
from contrastive_steering_base import generate_and_steer_batch_contrastive
from activation_pca import get_contrastive_activation_pca_
from CONFIGS.config_truthfulqa import Config
import torch
from evaluate_truthful_qa import evaluate_generation
from DATA.dataset_truthfulqa import prepare_dataset
import argparse
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torch.utils.data.distributed import DistributedSampler



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
    return parser.parse_args()

def main(model_path, save_dir,
         source_layer=14, target_layer=14, steering_strength=1):
    
    # Initialize accelerator
    accelerator = Accelerator()
    # Set seed for reproducibility across processes
    set_seed(42)

    # 0. cfg
    cfg = {}
    model_name = os.path.basename(model_path)
    # Create config
    cfg = Config(model_path=model_path,     
                model_name=model_name,
                save_dir=save_dir,
                source_layer=source_layer, target_layer=target_layer,
                steering_strength=steering_strength)

    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Process rank: {accelerator.process_index}")
        print(f"cfg: {cfg}")

    # 1. Load dataset and train-test split
    dataset = load_dataset("csv", data_files="DATA/TruthfulQA.csv")["train"]

    # train test split
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    dataset_train = split_dataset["train"]
    dataset_test = split_dataset["test"]

    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Train size: {len(dataset_train)}")
        print(f"Test size: {len(dataset_test)}")

    dataset_train =  dataset_train.select(range(cfg.n_train)) 
    dataset_test =  dataset_test.select(range(cfg.n_test)) 

    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Train size (SUBSAMPLE): {len(dataset_train)}")
        print(f"Test size (SUBSAMPLE): {len(dataset_test)}")



    # 2. Load model
    model_base = TransformerModelLoader(model_path, accelerator=accelerator)


    # 3. Generate Training Examples
    generate_and_steer_batch_contrastive(cfg, model_base, dataset_train, accelerator)

    print("done")





if __name__ == "__main__":
    args = parse_arguments()
    main(model_path=args.model_path,
         save_dir=args.save_dir,
         source_layer=args.source_layer, 
         target_layer=args.target_layer,
         steering_strength=args.steering_strength)

