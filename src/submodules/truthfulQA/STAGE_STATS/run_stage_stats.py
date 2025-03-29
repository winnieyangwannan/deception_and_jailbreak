import os
import sys
# get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_directory:", current_dir)
print(current_dir)
# get to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
os.chdir(parent_dir)
print("current_directory:", parent_dir)
print(parent_dir)
# append the parent directory to the system path
sys.path.append(parent_dir)


import argparse
import pickle
import csv


import math
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances

import plotly.graph_objects as go


from plotly.subplots import make_subplots
from plotly.figure_factory import create_quiver
import plotly.figure_factory as ff
import plotly.io as pio
import json

from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.distance import cdist
from scipy import stats
from CONFIGS.config_truthfulqa import Config
from STAGE_STATS.stage_statistics import get_state_quantification

def main(model_path, save_dir):
    # 0. cfg
    model_name = os.path.basename(model_path)
    # Create config
    cfg = Config(model_path=model_path,     
                model_name=model_name,
                save_dir=save_dir)
    
        
    # Open file
    file_path = os.path.join(
        "/home/winnieyangwn/Output/truthful_qa",
        model_name,
        "activations",
        f"{model_name}_activations_train.pkl",
    )
    with open(file_path, "rb") as file:
        activations = pickle.load(file)


    get_state_quantification(
    cfg,
    activations["activations_positive"],
    activations["activations_negative"],
    activations["correct_choice"],
    save_plot=True,
    save_name="train",
    )


if __name__ == "__main__":
    
    main(model_path="meta-llama/Llama-3.1-8B-Instruct",
         save_dir="/home/winnieyangwn/Output")
