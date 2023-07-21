import sys
from attention_lense.data.get_data_pl import DataModule
from attention_lense.model.get_model import get_model
from attention_lense.lense.get_lense import get_lense
from attention_lense.lense.lenseA import LenseA
import torch
import transformer_lens.utils as utils
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_nodes", default=1, type=int)
parser.add_argument("--resume_step", default=0, type=int)
parser.add_argument("--mixed_precision", default=False, type=bool, help="whether to use mixed precision for training")
parser.add_argument("--checkpoint_mode", default="step", type=str, choices=["step", "loss"], help="whether to checkpoint on train loss decrease or training step number")
parser.add_argument("--num_steps_per_checkpoint", default=5, type=int, help="number of steps after which to checkpoint (only valid for checkpoint_mode='step')")
parser.add_argument("--checkpoint_dir", default="./", type=str, help="directory to store checkpoint files in")
parser.add_argument("--accumulate_grad_batches", default=1, type=int, help="controls how many steps to accumulate gradients over")
parser.add_argument("--reload_checkpoint", default=None, type=str, help="path to checkpoint file, if set training resumes using this checkpoint")
parser.add_argument("--stopping_delta", default=1e-7, type=float, help="early stopping delta, if train loss decreases by <= delta we stop training")
parser.add_argument("--stopping_patience", default=2, type=int, help="number of checks with no improvement after which to stop training")
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
      #print("cached head: ", cache[self.hook_id].shape)
      inputs.append(cache[self.hook_id])
      #print("inputs: ", len(inputs))
      input_tensor = torch.stack(inputs)
      #print("input_tensor: ", input_tensor.shape)

      attn_lens_out = self.attn_lens(input_tensor)
      lens_out = attn_lens_out[0]
      #print("attn_lens_out: ", attn_lens_out.shape)
      #print("lens_out: ", lens_out.shape)

      return lens_out

  def kl_loss(self, logits, lens_logits):
    #print("logits: ", logits[:,-1,:].shape)
    #print("lens_out: ", lens_logits[:,-1,:].shape)
    kldiv = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    k_logits, k_lens_out = F.log_softmax(logits[:,-1,:], dim=-1), F.log_softmax(lens_logits[:,-1,:], dim=-1)

    #print("k_logits: ", k_logits.shape)
    #print("k_lens_out: ", k_lens_out.shape)
    loss = kldiv(k_lens_out, k_logits)
    return loss


  def training_step(self, train_batch, batch_idx):
      #x, y = train_batch
      #print("train step device: ", self.device)
      #self.model = get_model(device=self.device)
      prompt = train_batch['text']
      tokens = self.model.to_tokens(prompt)
      #print("prompt shape: ", tokens.shape)
      #print('device: ', self.device)
      #print('Tokens device number: ', tokens.get_device())
      #print('LLM device number: ', self.model.device)

      #self.model = get_model(device=self.device)

      with torch.no_grad():
          # only cache required hooks for lens
          logits, cache = self.model.run_with_cache(tokens, names_filter=self.hook_id, remove_batch_dim=False)
      #print("computed grads")

      # print("cache: ", cache)
      lens_logits = self.forward(cache)
      loss = self.kl_loss(logits, lens_logits)
      # print("loss: ", loss)
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

file_tag = f"attnlens-layer-{args.layer_number}"
early_stop_callback = EarlyStopping(monitor="train_loss", mode="min", min_delta=args.stopping_delta, patience=args.stopping_patience)
train_loss_checkpoint = ModelCheckpoint(
    save_top_k=10,
    monitor="train_loss",
    mode="min",
    dirpath=args.checkpoint_dir,
    filename=file_tag+"-{epoch:02d}-{train_loss:.2f}",
)
step_checkpoint = ModelCheckpoint(
    save_top_k=10,
    every_n_train_steps=args.num_steps_per_checkpoint,
    monitor="step",
    mode="max",
    dirpath=args.checkpoint_dir,
    filename=file_tag+"-{epoch:02d}-{step}",
)

checkpoint_callback = train_loss_checkpoint if args.checkpoint_mode == "loss" else step_checkpoint
training_precision = "16-mixed" if args.mixed_precision else 32

model = LightningLens()
data_module = DataModule()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', accelerator=accelerator,
                    precision=training_precision,
                     max_epochs=1,
                     num_nodes=args.num_nodes,
                     default_root_dir=args.checkpoint_dir,
                     accumulate_grad_batches=args.accumulate_grad_batches,
                     callbacks=[early_stop_callback, checkpoint_callback])
                     #TODO(MS): eventually use the profile to find bottlenecks: profiler='simple')

trainer.fit(model, data_module, ckpt_path=args.reload_checkpoint)
#TODO (MS): implement checkpointing
