import torch.types

from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(
    model_name: str = "gpt2", device: Union[str, torch.types.Device] = "cuda"
) -> AutoModelForCausalLM:
    """Loads and returns a pre-trained lens from the TransformerLens library by the given ``name``.

    Args:
        model_name (str): The name of the pre-trained lens.
        device (Union[str, torch.types.Device]): The device to train on.

    Examples:
        >>> lens = get_model(lens="gpt2")

    Returns:
        The pre-trained lens with hooks.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    print("lens created on device: ", device)
    return model, tokenizer
