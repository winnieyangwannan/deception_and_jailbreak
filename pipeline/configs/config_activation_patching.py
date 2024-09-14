import os

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    model_path: str
    model_alias: str
    task_name: str
    save_dir: str
    contrastive_type: str
    answer_type: str
    do_sample: bool
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 10
    n_test: int = 50
    batch_size: int = 5
    max_new_tokens: int = 5
    intervention: str = "no_intervention"
    transformer_lens: bool = True
    cache_pos: str = "prompt_all"  # "prompt_last_token" or "all" or "prompt_all"
    cache_name: str = "resid_pre"
    clean_type: str = "positive"
    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_dir, "runs", "activation_pca", self.model_alias)
