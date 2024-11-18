import os

from dataclasses import dataclass
from typing import Tuple

from sympy.physics.units import temperature


@dataclass
class Config:
    model_path: str
    model_alias: str
    task_name: str
    save_dir: str
    contrastive_type: str
    answer_type: str
    do_sample: bool
    framework: bool
    temperature: float
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 50  # 100
    n_test: int = 50  # 50
    batch_size: int = 5  # 5
    max_new_tokens: int = 100  # 100
    intervention: str = "no_intervention"
    cache_pos: str = "prompt_last_token"
    cache_name: str = "resid_post"

    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        temperature = self.temperature
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(
            save_dir, "runs", "activation_pca", self.model_alias, str(temperature)
        )
