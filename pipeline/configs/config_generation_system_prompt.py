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
    contrastive_type: Tuple[str]
    answer_type: Tuple[str]
    do_sample: bool
    framework: str
    temperature: float
    system_prompt_id_s: int
    system_prompt_id_e: int
    system_prompt_id: int = 0
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
        system_prompt_id = self.system_prompt_id
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(
            save_dir,
            "runs",
            "activation_pca",
            self.model_alias,
            "system_prompt_" + str(system_prompt_id),
        )
