import sys
sys.path.append('../')
from data.get_data import get_data
from model.get_model import get_model
from lense.get_lense import get_lense
from lense.lenseA import LenseA
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F
from tqdm import tqdm

#single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory=False
if device!="cpu":
    pin_memory=True

model = get_model(device=device)

# do the training one layer at a time to prevent ram from running out
hook_name = 'result'

n_layer = 9

#Initalize lense with model unembed/bias matrix
lens_param = {'unembed': model.W_U, 'bias': model.b_U, 'n_head': model.cfg.n_heads, 'd_model': model.cfg.d_model, 'd_vocab': model.cfg.d_vocab, 'lense_class': LenseA}
attn_lens = get_lense(n_layers=1, **lens_param).to(device)

name = "../train/attn_lens_layer_0" #for now we hold the lense constant
attn_lens = torch.load(name)

hook_id = utils.get_act_name(hook_name, n_layer)


prompt = "George Washington fought in the"
print("Prompt: ", prompt)
tokens = model.to_tokens(prompt)

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    
inputs = []
inputs.append(cache[hook_id])
input_tensor = torch.stack(inputs)

head = 8
layer_0_head = attn_lens.lenses[0].linears[head]
projected = layer_0_head(inputs[0][0][-1][head])

topk_token_vals, topk_token_preds = torch.topk(projected, 70)
print(model.to_string(topk_token_preds))
