import os
import sys
# get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current_directory:", current_dir)
print(current_dir)
from activation_pca import get_contrastive_activation_dim_reduction


# get to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# os.chdir(parent_dir)
# print("current_directory:", parent_dir)
# print(parent_dir)
# append the parent directory to the system path
sys.path.append(parent_dir)

import pickle


# model_path ="meta-llama/Llama-3.2-1B-Instruct"
model_path ="meta-llama/Llama-3.2-1B-Instruct"

save_dir = "/home/winnieyangwn/Output"
task_name = "truthful_qa" #sandbag
ds_method = "umap" # pca # umap
n_neighbors=10
min_dist=0.1
init="pca"

if task_name == "sandbag":
    from CONFIGS.config_sandbag import Config
elif task_name == "truthful_qa":
    from CONFIGS.config_truthfulqa import Config


# Create config
model_name = os.path.basename(model_path)

cfg = Config(model_path=model_path,     
            model_name=model_name,
            save_dir=save_dir)

# Open activation
file_path = os.path.join(
    "/home/winnieyangwn/Output",
    task_name,
    model_name,
    "activations",
    f"{model_name}_activations_train.pkl",
)
with open(file_path, "rb") as file:
    activations = pickle.load(file)



# Do dimensionality reduction
get_contrastive_activation_dim_reduction(cfg, activations, split="train",
label_type="model_choice", ds_method=ds_method, 
n_neighbors=n_neighbors, min_dist=min_dist, init=init)

get_contrastive_activation_dim_reduction(cfg, activations, split="train",
label_type="correct_choice", ds_method=ds_method,
 n_neighbors=n_neighbors, min_dist=min_dist, init=init)

