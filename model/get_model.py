from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

def get_model(model_name="gpt2_small", device="cuda"):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    return model
