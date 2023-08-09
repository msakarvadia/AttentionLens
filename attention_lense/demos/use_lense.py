import sys
sys.path.append('../')
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
import argparse


#### SET UP USER ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--lense_loc", default="/lus/grand/projects/SuperBERT/mansisak/extracted_lenses/gpt2-small/attnlens-layer-9-epoch=00-step=2795-train_loss=0.03.ckpt", type=str, help="path to dir containing all latest ckpts for a model")
parser.add_argument("--model", default="gpt2-small", choices=["gpt2-small","gpt2-large"], type=str, help="model that the lense corresponds to")
parser.add_argument("--layer_num", default=9, type=int, help="layer number that lense corresponds to")
args = parser.parse_args()

#single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

#get model
model = get_model(args.model,device=device)

attn_lens = torch.load(args.lense_loc)

prompts = ["George Washington fought in the",
            "The first president of the united states fought in the",
            "St. Peter's Bascillica is in the city of",
            "The leader of the United States live in the"
            ]

def interpret_layer(prompt, attn_lens):
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        
    inputs = []
    hook_name = 'result'
    hook_id = utils.get_act_name(hook_name, args.layer_num)
    inputs.append(cache[hook_id])
    input_tensor = torch.stack(inputs)

    for head in range(12):
        #head = 8
        print("Head: ", head)
        layer_head = attn_lens.lenses[0].linears[head]
        projected = layer_head(inputs[0][0][-1][head])

        topk_token_vals, topk_token_preds = torch.topk(projected, 70)
        print(model.to_string(topk_token_preds))
        print("______________________")

for prompt in prompts:
    print("Prompt: ", prompt)
    interpret_layer(prompt, attn_lens)
