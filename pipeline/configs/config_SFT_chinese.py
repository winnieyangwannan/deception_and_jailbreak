import os

from dataclasses import dataclass
from typing import Tuple

from sympy.physics.units import temperature

#


@dataclass
class Config:
    model_path: str
    model_alias: str
    model_path_big: str
    model_path_small: str
    model_name_small: str
    model_name_big: str
    data_dir: str
    save_dir: str
    do_sample: bool = True
    temperature: float = 1.0
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    batch_size: int = 512  # 512  # 5
    max_new_tokens: int = 100
    framework: str = "transformers"
    task_name: str = "deception_sft"
    contrastive_type: tuple = ("honest", "lying")
    answer_type: tuple = ("harmless", "harmful")

    # 100

    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # temperature = self.temperature
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(save_dir, "runs", "SFT_chinese", self.model_name_small)
