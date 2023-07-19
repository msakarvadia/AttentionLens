import sys
from attention_lense.data.get_data_pl import DataModule
from attention_lense.model.get_model import get_model
from attention_lense.lense.get_lense import get_lense
from attention_lense.lense.lenseA import LenseA
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import math
import argparse


#TODO (MS): clean up args and keep only relavent ones
#NOTE(MS): I copied these args from a different project so they might not be relavent anymore
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
parser.add_argument("--num_nodes", default=1, type=int)
parser.add_argument("--resume_step", default=0, type=int)
parser.add_argument("--num_steps_per_checkpoint", default=5, type=int)
parser.add_argument("--checkpoint_dir", default="/grand/projects/SuperBERT/mansisak/kd_ckpts/", type=str, help="directory to store checkpoint files in")
parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="controls how many steps to accumulate gradients over")
parser.add_argument("--reload_checkpoint", default=None, type=str, help="path to checkpoint file, if set training resumes using this checkpoint")
parser.add_argument("--layer_number", default=0, type=int)
args = parser.parse_args()

#### Pytorch lighting

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
    self.attn_lens = get_lense(n_layers=1, **lens_param)# .to(device)
   
  def setup(self, stage):
    #print("setup step device: ", self.device)
    #print("setup step work around device: ", self.trainer.strategy.root_device)
    self.model = get_model(device=self.trainer.strategy.root_device)
    return
    

  def forward(self, cache):
      #print("forward step device: ", self.device)
        
      inputs = []
      inputs.append(cache[self.hook_id])
      input_tensor = torch.stack(inputs)

      attn_lens_out = self.attn_lens(input_tensor)
      lens_out = attn_lens_out[0]

      return lens_out

  def kl_loss(self, logits, lens_logits):
    kldiv = torch.nn.KLDivLoss(reduction='batchmean')
    k_logits, k_lens_out = F.log_softmax(logits, dim=-1), F.log_softmax(lens_logits, dim=-1)

    loss = kldiv(k_lens_out, k_logits)
    return loss


  def training_step(self, train_batch, batch_idx):
      #x, y = train_batch
      #print("train step device: ", self.device)
      #self.model = get_model(device=self.device)
      prompt = train_batch['text']
      tokens = self.model.to_tokens(prompt)
      #print('device: ', self.device)
      #print('Tokens device number: ', tokens.get_device())
      #print('LLM device number: ', self.model.device)

      #self.model = get_model(device=self.device)

      with torch.no_grad():
          # only cache required hooks for lens
          logits, cache = self.model.run_with_cache(tokens, names_filter=self.hook_id, remove_batch_dim=False)
      #print("computed grads")

      lens_logits = self.forward(cache)
      loss = self.kl_loss(logits, lens_logits)
      self.log('train_loss', loss)
      return loss


  '''
  def validation_step(self, val_batch, batch_idx):
      x, y = val_batch
      logits = self.forward(x)
      loss = self.cross_entropy_loss(logits, y)
      self.log('val_loss', loss)
  '''

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
  
  #TODO(MS): register an early stopping call back which quits training if the loss/some metric drops below a certain pont
  #TODO(MS): when training quits, save a copy of the appropriately named lense
  #TODO(MS): test and make sure distributed training works accross nodes

#train
#LLM = get_model()
print(args.checkpoint_dir)
print(args.accumulate_grad_batches)
print(args.reload_checkpoint)
model = LightningLens()
data_module = DataModule()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', accelerator=accelerator,
                     max_epochs=1,
                     num_nodes=args.num_nodes,
                     default_root_dir=args.checkpoint_dir,
                     accumulate_grad_batches=args.accumulate_grad_batches)
                     #TODO(MS): eventually use the profile to find bottlenecks: profiler='simple')

trainer.fit(model, data_module, ckpt_path=args.reload_checkpoint)
#TODO (MS): implement checkpointing
