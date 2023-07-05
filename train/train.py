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
import math
import argparse


#TODO (MS): clean up args and keep only relavent ones
#### SET UP USER ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--device", type=int)
parser.add_argument("--lr", default=5e-5, type=float)
parser.add_argument("--temp", default=3, type=float)
parser.add_argument("--alpha", default=0.5, type=float)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--warmup_steps", default=10000, type=int)
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--resume_step", default=0, type=int)
parser.add_argument("--num_steps_per_checkpoint", default=5, type=int)
parser.add_argument("--checkpoint_dir", default="/grand/projects/SuperBERT/mansisak/kd_ckpts/", type=str)
parser.add_argument("--teacher_checkpoint", default="bert-base-uncased", type=str)
parser.add_argument("--student_checkpoint", default="distilbert-base-uncased", type=str)
parser.add_argument("--log_file", default="training.log", type=str)
parser.add_argument("--loss_file", default="training_loss.csv", type=str)
parser.add_argument("--layer_number", default=0, type=int)
args = parser.parse_args()

#single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
pin_memory=False
if device!="cpu":
    pin_memory=True

batch_size = args.batch_size
dataloader = get_data(streaming=True, 
                        dataset_name="c4",
                        batch_size=batch_size,
                        pin_memory=pin_memory,
                        device=device,
                        num_workers=16)

model = get_model(device=device)

# do the training one layer at a time to prevent ram from running out
hook_name = 'result'
kldiv = torch.nn.KLDivLoss(reduction='batchmean')

#TODO (MS): implement checkpointing
n_layer = args.layer_number

#Initalize lense with model unembed/bias matrix
lens_param = {'unembed': model.W_U, 'bias': model.b_U, 'n_head': model.cfg.n_heads, 'd_model': model.cfg.d_model, 'd_vocab': model.cfg.d_vocab, 'lense_class': LenseA}
attn_lens = get_lense(n_layers=1, **lens_param).to(device)
hook_id = utils.get_act_name(hook_name, n_layer)
total_steps = 5
progress_bar = tqdm(range(total_steps))

for i, data in enumerate(dataloader):
  if i == total_steps:
      break

  prompt = data['text']
  tokens = model.to_tokens(prompt)

  with torch.no_grad():
      logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    
  inputs = []
  inputs.append(cache[hook_id])
  input_tensor = torch.stack(inputs)
   # print(input.requires_grad, input.grad_fn)f

    # lens_out = lens(input)
    # print(lens_out.requires_grad)
    # print(lens_out.grad_fn)

    #print("input size: ",input_tensor.size())
    #print("logits size: ",logits.size())

  attn_lens_out = attn_lens(input_tensor)

    #print("attenion lens original output size: ", attn_lens_out.size())

  lens_out = attn_lens_out[0]
    #print("lense output size: ", lens_out.size())

    #TODO (MS): are we supposed to log softmax both, or just one of these quantities
  k_logits, k_lens_out = F.log_softmax(logits, dim=-1), F.log_softmax(lens_out, dim=-1)

    #print(k_lens_out.requires_grad)
    #print(k_lens_out.grad_fn)

  loss = kldiv(k_lens_out, k_logits)
  loss.backward()
    
    #update tqdm bar
  progress_bar.update(1)
    
  #TODO: need to save attn lens in correct location
name = "attn_lens_layer_"+str(n_layer)
torch.save(attn_lens, name)
#TODO (MS): need to test that lense is useable for model analysis later
