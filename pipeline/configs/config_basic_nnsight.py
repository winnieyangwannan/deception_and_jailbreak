import os

from dataclasses import dataclass
from typing import Tuple

from sympy.physics.units import temperature


@dataclass
class Config:
    model_path: str
    model_alias: str
    save_dir: str
    task_name: str = "nnsight_basic"
    contrastive_type: tuple = ("honest", "lying")
    answer_type: tuple = ("yes", "no")
    do_sample: bool = True
    temperature: int = 1.0
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 100  # 100
    n_test: int = 50  # 50
    batch_size: int = 256  # 256
    max_new_tokens: int = 100  # 100

    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)

        return os.path.join(save_dir, "runs", "activation_pca", self.model_alias)
