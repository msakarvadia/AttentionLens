import torch.types

from typing import Union
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)


def get_model(
    model_name: str = "gpt2-small", device: Union[str, torch.types.Device] = "cuda"
) -> HookedTransformer:
    """Loads and returns a pre-trained model from the HookedTransformer library by the given ``name``.

    Args:
        model_name (str): The name of the pre-trained model.
        device (Union[str, torch.types.Device]): The device to train on.

    Examples:
        >>> model = get_model(model="gpt2-small")

    Returns:
        The pre-trained model with hooks.
    """
    # model = HookedTransformer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True

    print("model created on device: ", device)
    return model
