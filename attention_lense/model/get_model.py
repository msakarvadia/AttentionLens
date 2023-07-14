from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

def get_model(model_name="gpt2-small", device="cuda"):
    #model = HookedTransformer.from_pretrained(model_name)
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.cfg.use_attn_result = True
    
    print("model created on device: ", device)
    return model
