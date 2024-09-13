import torch
from sae_lens import HookedSAETransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import transformer_lens.utils as utils


def load_model(cfg):
    model_path = cfg.model_path
    print("loading model")
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", end="\n\n")

    if cfg.transformer_lens:
        print("loading model from transformer_lens")
        if cfg.dtype == "bfloat16":
            model = HookedSAETransformer.from_pretrained(
                model_path,
                device=device,
                dtype=torch.bfloat16,
                fold_ln=True,
                # center_unembed=False,
                # center_writing_weights=False,
                # refactor_factored_attn_matrices=False,
            )
        else:
            model = HookedSAETransformer.from_pretrained(
                model_path,
                device=device,
                fold_ln=True,
                # center_unembed=False,
                # center_writing_weights=False,
                # refactor_factored_attn_matrices=False,
            )
        model.tokenizer.padding_side = "left"

    else:
        print("loading model from transformers")
        model = TransformerModelLoader(cfg)

    print(
        "free(Gb):",
        torch.cuda.mem_get_info()[0] / 1000000000,
        "total(Gb):",
        torch.cuda.mem_get_info()[1] / 1000000000,
    )
    return model


class TransformerModelLoader:
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=cfg.dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        self.model.requires_grad_(False)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # config

        self.cfg = HookedTransformerConfig.unwrap(self.model.config)
        self.cfg.tokenizer_prepends_bos = len(self.tokenizer.encode("")) > 0

        self.model_block_modules = self._get_model_block_modules(cfg)

    def _get_model_block_modules(self, cfg):
        if "llama" in cfg.model_path:
            return self.model.model.layers
        if "gpt" in cfg.model_path:
            return self.model.transformer.h

    def to_single_token(self, input_token):
        token = self.tokenizer(
            input_token,
            return_tensors="pt",
            padding=True,
        )["input_ids"]
        if self.cfg.tokenizer_prepends_bos:
            single_token = utils.get_tokens_with_bos_removed(self.tokenizer, token).to(
                self.model.device
            )
        else:
            single_token = token
        return single_token
