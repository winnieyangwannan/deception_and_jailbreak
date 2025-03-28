import contextlib
import functools
import gc
import io
import textwrap
from typing import Callable, List, Tuple

import einops
import numpy as np
import pandas as pd
import requests
import torch
from colorama import Fore

from datasets import load_dataset
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split
from torch import Tensor
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from accelerate import Accelerator
from transformers import BitsAndBytesConfig



class TransformerModelLoader:
    def __init__(self, model_path, accelerator=None):

        self.model_path = model_path

        if torch.cuda.is_available(): 
            # Quantization config (4-bit)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,         # improves performance a bit
                bnb_4bit_quant_type="nf4",              # "nf4" = best for LLMs
                bnb_4bit_compute_dtype=torch.float16    # can also use torch.bfloat16
            )
            # load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                ).eval()

            self.model.requires_grad_(False)

        # load tokenizer based on process
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # model_block
        self.model_block_modules = self._get_model_block_modules()



    def _get_model_block_modules(self):
        # this will be used when creating hooks

        if "gpt" in self.model_path:
            return self.model.transformer.h
        elif "Llama" in self.model_path:
            return self.model.model.layers
        elif "gemma" in self.model_path:
            return self.model.model.layers
        elif "Qwen-" in self.model_path:
            return self.model.transformer.h
        elif "Qwen2" in self.model_path:
            return self.model.model.layers
        elif "Yi" in self.model_path:
            return self.model.model.layers
