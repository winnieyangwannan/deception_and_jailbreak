import os
import pickle

model_alias = "Meta-Llama-3-8B-Instruct"

score_path = os.path.join(
    "D:\Data\deception",
    "runs",
    "activation_pca",
    model_alias,
    "activation_steering",
    "positive_addition",
    "stage_stats",
)

cos_all = []
for ii in range(32):
    with open(
        score_path + os.sep + f"{model_alias}_stage_stats_source_{ii}_target_{ii}.pkl",
        "rb",
    ) as f:
        data = pickle.load(f)
    cos = data["stage_3"]["cos_positive_negative"]
    cos_all.append(cos)


data_save = {}
data_save["cos_all"] = cos_all
with open(score_path + os.sep + f"{model_alias}_cos_sim_all_layers.pkl", "wb") as f:
    pickle.dump(data_save, f)
