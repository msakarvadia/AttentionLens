import sys
sys.path.append('../')
from attention_lense.model.get_model import get_model
from attention_lense.model.get_model import get_model
from attention_lense.lense.get_lense import get_lense
from attention_lense.lense.lenseA import LenseA
#from attention_lense.train.train_pl import LightningLens
#import attention_lense.train as AL
#import pytorch_lightning as pl
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F
import glob
import os

#single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device="cpu"

model = get_model(device=device)

# do the training one layer at a time to prevent ram from running out
hook_name = 'result'



#Initalize lense with model unembed/bias matrix
lens_param = {'unembed': model.W_U, 'bias': model.b_U, 'n_head': model.cfg.n_heads, 'd_model': model.cfg.d_model, 'd_vocab': model.cfg.d_vocab, 'lense_class': LenseA}
attn_lens = get_lense(n_layers=1, **lens_param).to(device)

name = "../train/attn_lens_layer_0" #for now we hold the lense constant
name = "/lus/eagle/projects/datascience/mansisak/AttentionLens/attention_lense/checkpoint/attnlens-layer-9-epoch=00-step=520-train_loss=0.17.ckpt"
name = "/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/gpt2-small/ckpt_0/attnlens-layer-0-epoch=00-step=590-train_loss=0.16.ckpt"

#Iterate through all ckpts
# accessing and printing files in directory and subdirectory
ckpt_dir = "/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/gpt2-small/"

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

def extract_and_save_lense_from_ckpt(ckpt_filepath, save_filepath="sample_lense"):
    attn_lens_cls = torch.load(ckpt_filepath, map_location='cpu')
    a = attn_lens_cls["state_dict"]
    for key in list(a.keys()):
        if not key.startswith("attn_lens"):
           del a[key]
        else:
            change_dict_key(a, key, key[10:])

    attn_lens.load_state_dict(a) 
    torch.save(attn_lens, save_filepath)


def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir="gpt2_small"):
    for filename in glob.glob(ckpt_dir+'**/*.ckpt', recursive=True):
        #print(filename)  # print file name
        print(save_dir+"/"+os.path.basename(filename))  # print file name
        save_filepath = save_dir+"/"+os.path.basename(filename)

        #if save_dir doesn't exist, create it
        # Check whether the specified path exists or not
        isExist = os.path.exists(save_dir)
        if not isExist:
           # Create a new directory because it does not exist
           os.makedirs(save_dir)

        extract_and_save_lense_from_ckpt(filename, save_filepath=save_filepath)

iter_thru_ckpts_extract_lenses(ckpt_dir)

extract_and_save_lense_from_ckpt(ckpt_filepath=name, save_filepath="sample_lense")

attn_lense = torch.load("sample_lense")

#TODO Update layer:
n_layer = 0
hook_id = utils.get_act_name(hook_name, n_layer)

prompt = "George Washington fought in the"
print("Prompt: ", prompt)
tokens = model.to_tokens(prompt)

with torch.no_grad():
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    
inputs = []
inputs.append(cache[hook_id])
input_tensor = torch.stack(inputs)

for head in range(12):
    #head = 8
    print("Head: ", head)
    layer_0_head = attn_lens.lenses[0].linears[head]
    projected = layer_0_head(inputs[0][0][-1][head])

    topk_token_vals, topk_token_preds = torch.topk(projected, 70)
    print(model.to_string(topk_token_preds))
    print("______________________")
