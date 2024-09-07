import torch
from sae_lens import HookedSAETransformer


def load_model(cfg):
    model_path = cfg.model_path
    print("loading model")
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

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
    print(
        "free(Gb):",
        torch.cuda.mem_get_info()[0] / 1000000000,
        "total(Gb):",
        torch.cuda.mem_get_info()[1] / 1000000000,
    )
    return model
