import os

from dataclasses import dataclass
from typing import Tuple

from sympy.physics.units import temperature


@dataclass
class Config:
    model_path: str
    model_alias: str
    model_path_before: str
    model_alias_before: str
    save_dir: str
    lora: bool
    checkpoint: int
    task_name: str = "sft_to_lie"
    contrastive_type: tuple = ("honest", "lying")
    answer_type: tuple = ("yes", "no")
    do_sample: bool = True
    framework: str = "transformers"
    temperature: int = 1.0
    dtype: str = "bfloat16"  # "bfloat16" or "default"
    n_train: int = 20  # 100
    n_test: int = 50  # 50
    batch_size: int = 256  # 256
    max_new_tokens: int = 100  # 100
    intervention: str = "no_intervention"
    cache_pos: str = "prompt_last_token"
    cache_name: str = "resid_post"

    # for generation_trajectory

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        if self.checkpoint:

            return os.path.join(
                save_dir,
                "runs",
                self.task_name,
                f"lora_{self.lora}",
                self.model_alias,
                str(self.checkpoint),
            )
        else:
            return os.path.join(
                save_dir, "runs", self.task_name, f"lora_{self.lora}", self.model_alias
            )

    def checkpoint_path(self) -> str:
        "D:\Data\deception\sft\sft_to_lie\Yi-6B-Chat\lora_True\checkpoint-100"
        return os.path.join(
            self.save_dir,
            "sft",
            self.task_name,
            self.model_alias_before,
            f"lora_{self.lora}",
            f"checkpoint-{self.checkpoint}",
        )
