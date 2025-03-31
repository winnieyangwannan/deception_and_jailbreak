from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import pandas as pd
import os
from MODEL.model_base import TransformerModelLoader
from contrastive_steering_truthfulqa import generate_and_steer_batch_contrastive
from CONFIGS.config_truthfulqa_sft import Config
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
from DATA.dataset_truthfulqa import get_correct_choice


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
        default="winnieyangwannan/sft_to_lie_Yi-6B-Chat_1",
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
         source_layer=14, target_layer=14, steering_strength=1,
         ):
    
    # Initialize accelerator
    accelerator = Accelerator()
    # Set seed for reproducibility across processes
    set_seed(42)

    model_name = os.path.basename(model_path)
    model_name_before = model_name.split("_")[-2]
    if "gemma" in model_name_before:
        model_path_before = "google/" + model_name_before
    elif "Llama" in model_name_before:
        model_path_before = "meta-llama/" + model_name_before
    elif "Qwen" in model_name_before:
        model_path_before = "Qwen/" + model_name_before
    elif "Yi" in model_name_before:
        model_path_before = "01-ai/" + model_name_before
    
    checkpoint = int(model_name.split("_")[-1])

    # 0. cfg
    cfg = {}
    # Create config
    cfg = Config(model_path=model_path, 
                model_path_before=model_path_before,    
                model_name=model_name,
                model_name_before=model_name_before,
                save_dir=save_dir,
                source_layer=source_layer, target_layer=target_layer,
                steering_strength=steering_strength,
                checkpoint=checkpoint)

    # Only log from main process to avoid duplicate output
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs")
        print(f"Process rank: {accelerator.process_index}")
        print(f"cfg: {cfg}")

    # 1. Load dataset and train-test split
    if accelerator.is_main_process:
        print("Loading dataset...")
    dataset = load_dataset("csv", data_files=f"DATA/Llama-3.1-8B-Instruct_truthfulQA.csv")["train"]


    # train test split
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    dataset_train = split_dataset["train"]
    dataset_test = split_dataset["test"]

    # Only log from main process to avoid duplicate output
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

    dataset_train = get_correct_choice(dataset_train)
    dataset_test = get_correct_choice(dataset_test)

    # model_choice_positive_all, model_choice_negative_all = evaluate_generation(cfg, train_test="train")

    # 2. Load model
    if accelerator.is_main_process:
        print("Loading model...")
    model_base = TransformerModelLoader(cfg, accelerator=accelerator)

    # 3. Generate and Cache Training Examples
    generate_and_steer_batch_contrastive(cfg, model_base, dataset_train, accelerator,
                                         steering_method="cache_pos",
                                         train_test="train", do_pca=True, no_system=True)


if __name__ == "__main__":
    args = parse_arguments()
    main(model_path=args.model_path,
         save_dir=args.save_dir,
         source_layer=args.source_layer, 
         target_layer=args.target_layer,
         steering_strength=args.steering_strength,
         )

