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
    category: str 
    n_positions: int = 1
    n_train: int = 512 #512
    n_test: int = 128 #128
    batch_size: int = 16 ##32  # 128  # 5
    max_new_tokens: int = 100 #100
    task_name: str = "sandbag"
    steer_type: str = "negative_addition"  # addition
    source_layer: int = 14  # "last"
    target_layer: int = 14  # "last"
    contrastive_type: tuple = ("control", "sandbag")
    steering_strength: float = 1
    # label_type: tuple = ("A", "B", "C", "D")
    label_type: tuple = ("A", "B")
    lora: bool = True
    checkpoint: int = 1

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        task_name = self.task_name
        category = self.category
        steer_type = self.steer_type
        source_layer = self.source_layer
        steering_strength = self.steering_strength
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(
            save_dir, task_name, category, self.model_name
        )
    

        