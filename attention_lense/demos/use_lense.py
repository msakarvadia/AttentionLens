import sys
sys.path.append('../')
from attention_lense.model.get_model import get_model
from attention_lense.model.get_model import get_model
from attention_lense.lense.get_lense import get_lense
from attention_lense.lense.lenseA import LenseA
#from attention_lense.train.train_pl import LightningLens
#import attention_lense.train as AL
import pytorch_lightning as pl
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F
from tqdm import tqdm

#single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory=False
if device!="cpu":
    pin_memory=True

#Lightning Lens
class LightningLens(pl.LightningModule):

  def __init__(self):
    super().__init__()

    #self.model=model
    #print("init step device: ", self.device)
    self.base_model = get_model(device=self.device)
    self.hook_name = 'result'
    self.n_layer = 0
    self.hook_id = utils.get_act_name(self.hook_name, self.n_layer)

    #Initalize lense with model unembed/bias matrix
    lens_param = {'unembed': self.base_model.W_U, 'bias': self.base_model.b_U, 'n_head':self.base_model.cfg.n_heads, 'd_model': self.base_model.cfg.d_model, 'd_vocab': self.base_model.cfg.d_vocab, 'lense_class': LenseA}

    #making lense
    self.attn_lens = get_lense(n_layers=1, **lens_param)



model = get_model(device=device)

# do the training one layer at a time to prevent ram from running out
hook_name = 'result'

n_layer = 9

#Initalize lense with model unembed/bias matrix
lens_param = {'unembed': model.W_U, 'bias': model.b_U, 'n_head': model.cfg.n_heads, 'd_model': model.cfg.d_model, 'd_vocab': model.cfg.d_vocab, 'lense_class': LenseA}
attn_lens = get_lense(n_layers=1, **lens_param).to(device)

name = "../train/attn_lens_layer_0" #for now we hold the lense constant
name = "/lus/eagle/projects/datascience/mansisak/AttentionLens/attention_lense/checkpoint/attnlens-layer-9-epoch=00-step=520-train_loss=0.17.ckpt"
#attn_lens = torch.load(name)
#attn_lens = LightningLens()
attn_lens_cls = torch.load(name)
print(type(attn_lens_cls))
print(attn_lens_cls.keys())
print(type(attn_lens_cls['state_dict']))
print(attn_lens_cls['state_dict'].keys())
a = attn_lens_cls["state_dict"]

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

d = {'k1': 1, 'k2': 2, 'k3': 3}

for key in list(a.keys()):
    if not key.startswith("attn_lens"):
       del a[key]
    else:
        change_dict_key(a, key, key[10:])

print("MODIFIED STATE DICT")
print(a.keys())


attn_lens.load_state_dict(a) 
print("LOADED LENS")

print(type(attn_lens))
print((attn_lens))


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
