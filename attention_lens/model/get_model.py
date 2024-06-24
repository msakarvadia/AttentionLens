import torch.types

from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(
    model_name: str = "gpt2", device: Union[str, torch.types.Device] = "cuda"
) -> AutoModelForCausalLM:
    """Loads and returns a model and tokenizer from the modified Hugging Face Transformers library.

    Args:
        model_name (str): The name of the pre-trained model.
        device (Union[str, torch.types.Device]): The device to train on.

    Examples:
        >>> model, tokenizer = get_model("gpt2")

    Returns:
        The light-weight hooked model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    print("Model initialized on device: ", device)
    return model, tokenizer
