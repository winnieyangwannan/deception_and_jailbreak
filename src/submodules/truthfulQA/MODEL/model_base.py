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
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM



class TransformerModelLoader:
    def __init__(self, cfg, accelerator=None):

        self.model_path = cfg.model_path
        
        if cfg.lora:

            base_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path_before,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            peft_model = PeftModel.from_pretrained(
                base_model,
                cfg.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            if cfg.checkpoint:
                print("********************************************")
                print("LOADING MODEL FROM CHECKPOINT")
                print("********************************************")

                # load from checkpoint
                active_adapters = peft_model.active_adapters
                active_adapter = active_adapters[0]
                peft_model.load_adapter(
                    cfg.model_path,
                    active_adapter,
                    is_trainable=True,
                )

            # merge lora adaptor with the base model and unload the LoRA model
            self.model = peft_model.merge_and_unload()
            # Set the model to evaluation mode
            self.model.eval()

        elif torch.cuda.is_available(): 
            # Quantization config (4-bit)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,         # improves performance a bit
                bnb_4bit_quant_type="nf4",              # "nf4" = best for LLMs
                bnb_4bit_compute_dtype=torch.float16    # can also use torch.bfloat16
            )
            # load model
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
            ).eval()

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto",
                ).eval()

            self.model.requires_grad_(False)


        # load tokenizer based on process
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
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
