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
import wandb
from pytorch_lightning.loggers import WandbLogger, CSVLogger



#TODO (MS): clean up args and keep only relavent ones
#NOTE(MS): I copied these args from a different project so they might not be relavent anymore
#### SET UP USER ARGS
parser = argparse.ArgumentParser()
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--epochs", default=3, type=int)
parser.add_argument("--max_ckpt_num", default=1, type=int, help="maximum number of ckpts to save")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_nodes", default=1, type=int)
parser.add_argument("--mixed_precision", default=True, type=bool, help="whether to use mixed precision for training")
parser.add_argument("--checkpoint_mode", default="step", type=str, choices=["step", "loss"], help="whether to checkpoint on train loss decrease or training step number")
parser.add_argument("--num_steps_per_checkpoint", default=5, type=int, help="number of steps after which to checkpoint (only valid for checkpoint_mode='step')")
parser.add_argument("--checkpoint_dir", default="./checkpoint/", type=str, help="directory to store checkpoint files in")
parser.add_argument("--accumulate_grad_batches", default=10, type=int, help="controls how many steps to accumulate gradients over")
parser.add_argument("--reload_checkpoint", default=None, type=str, help="path to checkpoint file, if set training resumes using this checkpoint")
parser.add_argument("--stopping_delta", default=1e-7, type=float, help="early stopping delta, if train loss decreases by <= delta we stop training")
parser.add_argument("--stopping_patience", default=2, type=int, help="number of checks with no improvement after which to stop training")
parser.add_argument("--layer_number", default=9, type=int)
args = parser.parse_args()

#### Logger
#wandb_logger = WandbLogger(log_model='all', project="attn_lens1")
#csv_logger = CSVLogger(save_dir=args.checkpoint_dir, name="attn_lense")

#### Pytorch lighting

class LightningLens(pl.LightningModule):

  def __init__(self):
    super().__init__()
    self.base_model = get_model(device=self.device)
    self.hook_name = 'result'
    self.n_layer = args.layer_number
    self.hook_id = utils.get_act_name(self.hook_name, self.n_layer)

    #Initalize lense with model unembed/bias matrix
    lens_param = {'unembed': self.base_model.W_U, 'bias': self.base_model.b_U, 'n_head':self.base_model.cfg.n_heads, 'd_model': self.base_model.cfg.d_model, 'd_vocab': self.base_model.cfg.d_vocab, 'lense_class': LenseA}

    #making lense
    self.attn_lens = get_lense(n_layers=1, **lens_param)# .to(device)
   
  def setup(self, stage):
    self.model = get_model(device=self.trainer.strategy.root_device)
    return
    
  def forward(self, cache):
      inputs = []
      inputs.append(cache[self.hook_id])
      input_tensor = torch.stack(inputs)

      attn_lens_out = self.attn_lens(input_tensor)
      lens_out = attn_lens_out[0]
      return lens_out

  def kl_loss(self, logits, lens_logits):
    kldiv = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    k_logits, k_lens_out = F.log_softmax(logits[:,-1,:], dim=-1), F.log_softmax(lens_logits[:,-1,:], dim=-1)

    loss = kldiv(k_lens_out, k_logits)
    return loss

  def training_step(self, train_batch, batch_idx):
      prompt = train_batch['text']
      tokens = self.model.to_tokens(prompt)

      with torch.no_grad():
          # only cache required hooks for lens
          logits, cache = self.model.run_with_cache(tokens, names_filter=self.hook_id, remove_batch_dim=False)

      lens_logits = self.forward(cache)
      loss = self.kl_loss(logits, lens_logits)
      self.log('train_loss', loss, prog_bar=True)
      #my_dict = {"train_loss":loss}
      #wandb.log(my_dict, step=trainer.global_step)
      return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
    return optimizer
  
  #TODO(MS): register an early stopping call back which quits training if the loss/some metric drops below a certain pont
  #TODO(MS): when training quits, save a copy of the appropriately named lense
  #TODO(MS): test and make sure distributed training works accross nodes


file_tag = f"attnlens-layer-{args.layer_number}"
early_stop_callback = EarlyStopping(monitor="train_loss", mode="min", min_delta=args.stopping_delta, patience=args.stopping_patience)

train_loss_checkpoint = ModelCheckpoint(
    save_top_k=args.max_ckpt_num,
    monitor="train_loss",
    mode="min",
    dirpath=args.checkpoint_dir,
    filename=file_tag+"-{epoch:02d}-{train_loss:.2f}",
)
step_checkpoint = ModelCheckpoint(
    save_top_k=args.max_ckpt_num,
    every_n_train_steps=args.num_steps_per_checkpoint,
    monitor="step",
    mode="max",
    dirpath=args.checkpoint_dir,
    filename=file_tag+"-{epoch:02d}-{step}",
)

checkpoint = ModelCheckpoint(
    #TODO change the max num of checkpoints
    save_top_k=args.max_ckpt_num,
    monitor="train_loss",
    mode="min",
    dirpath=args.checkpoint_dir,
    filename=file_tag+"-{epoch:02d}-{step}-{train_loss:.2f}",
    every_n_train_steps=args.num_steps_per_checkpoint,
)
logging_checkpoint = ModelCheckpoint(
    monitor="train_loss",
    mode="min",
)
latest_checkpoint = ModelCheckpoint(monitor='step', 
        mode='max', 
        every_n_train_steps=10, 
        save_top_k=1,dirpath=args.checkpoint_dir,
        filename="latest-{epoch}-{step}")

checkpoint_callback = train_loss_checkpoint if args.checkpoint_mode == "loss" else step_checkpoint
training_precision = "16-mixed" if args.mixed_precision else 32

#TODO hard coding a checkpoint for now
checkpoint_callback = checkpoint
print("Checkpoint Type: ", args.checkpoint_mode)

def getLens():
    return LightningLens()

model = getLens()

data_module = DataModule()
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(strategy='ddp_find_unused_parameters_true', accelerator=accelerator,
                    precision=training_precision,
                     max_epochs=1,
                     num_nodes=args.num_nodes,
                     default_root_dir=args.checkpoint_dir,
                     accumulate_grad_batches=args.accumulate_grad_batches,
                     callbacks=[early_stop_callback, checkpoint],
                     #callbacks=[early_stop_callback, logging_checkpoint, latest_checkpoint],
                    #flush_logs_every_n_steps=100,
                    #log_every_n_steps=1,
                    #logger=csv_logger)
                    #logger=wandb_logger)
                     #TODO(MS): eventually use the profile to find bottlenecks: profiler='simple')
                    )

### Automate reloading ckpt if ckpt_dir exists
reload_dir = args.reload_checkpoint
if (reload_dir == None):
    print("Ckpt dir to reload from is not specified, searching for existing ckpt dir.")


trainer.fit(model, data_module, ckpt_path=reload_dir)
