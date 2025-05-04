


path = "/home/winnieyangwn/Output/sandbag/sandbag/wmdp-bio/Llama-3.1-8B-Instruct/activations/Llama-3.1-8B-Instruct_activations_train.pkl"

import pickle

with open(path, 'rb') as f:
    data = pickle.load(f)

print(data)


data["activations_positive"][0].shape


data["negative_mean_act_1"][0].shape
