import os
from dataclasses import dataclass
from typing import Tuple
from sympy.physics.units import temperature



@dataclass
class Config:
    model_path: str
    model_path_before: str
    model_name: str
    model_name_before: str
    save_dir: str
    n_positions: int = 1
    n_train: int = 0
    n_test: int = 0
    batch_size: int = 128 #128
    max_new_tokens: int = 100
    task_name: str = "truthful_qa"
    steer_type: str = "negative_addition"  # addition
    source_layer: int = 14  # "last"
    target_layer: int = 14  # "last"
    contrastive_type: tuple = ("factual", "misconception")
    steering_strength: float = 1
    label_type: tuple = ("A", "B")
    lora: bool = True
    checkpoint: int = 1

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        task_name = self.task_name
        steer_type = self.steer_type
        source_layer = self.source_layer
        steering_strength = self.steering_strength
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(
            save_dir, task_name, self.model_name
        )
