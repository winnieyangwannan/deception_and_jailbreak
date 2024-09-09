from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
from torch import Tensor
from jaxtyping import Float, Int, Bool
import torch


def logits_to_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    positive_answer_token: Float[Tensor, "batch 1"],
    negative_answer_token: Float[Tensor, "batch 1"],
    per_prompt: bool = False,
) -> Union[Float[Tensor, ""], Float[Tensor, "batch"]]:
    """
    Get logit difference between the correct and incorrect answers
    :param logits:
    :param positive_answer_token: answer tokens of positive prompts
    :per_prompt: logit difference per prompt
    :return: a scaler for mean logit difference across samples
    """
    # (1) Get last token logits only
    logits = logits[:, -1, :]  # [batch, d_vocab]

    # (2) Get the index into correct dimension
    # squeeze because the index tensor must have the same number of dimensions as input tensor
    correct_index = positive_answer_token.unsqueeze(1)  # [batch, 1]
    incorrect_index = negative_answer_token[:].unsqueeze(1)  # [batch, 1]

    # (3) gather the logits corresponding to the indices
    correct_logits = logits.gather(dim=1, index=correct_index)
    incorrect_logits = logits.gather(dim=1, index=incorrect_index)
    logit_diff = correct_logits - incorrect_logits

    return logit_diff if per_prompt else logit_diff.mean()
