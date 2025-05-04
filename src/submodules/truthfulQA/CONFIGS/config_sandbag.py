import os
from dataclasses import dataclass
from typing import Tuple
from sympy.physics.units import temperature



@dataclass
class Config:
    model_path: str
    model_name: str
    save_dir: str
    category: str 
    n_positions: int = 1
    n_train: int = 4#512 #512
    n_test: int = 4#512 #128
    batch_size: int = 4#64 ##32  # 128  # 5
    max_new_tokens: int = 1#100 #100
    task_name: str = "sandbag"
    steer_type: str = "negative_addition"  # addition
    source_layer: int = 14  # "last"
    target_layer: int = 14  # "last"
    contrastive_type: tuple = ("control", "sandbag")
    steering_strength: float = 1
    # label_type: tuple = ("A", "B", "C", "D")
    label_type: tuple = ("A", "B")
    lora: bool = False
    checkpoint: int = 0
    system_type: str = "sandbag" # lie

    def artifact_path(self) -> str:
        save_dir = self.save_dir
        task_name = self.task_name
        category = self.category
        steer_type = self.steer_type
        source_layer = self.source_layer
        steering_strength = self.steering_strength
        system_type = self.system_type
        # return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", "activation_pca", self.model_alias)
        return os.path.join(
            save_dir, task_name, system_type, category, self.model_name
        )
    

        