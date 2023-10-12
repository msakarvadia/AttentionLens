import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
  """Initializes a DataLoader object for "bookcorpus". Support for more datasets coming soon.

  Returns:
      A dataloader.
  """

  def setup(self, stage):
    self.data  = load_dataset('bookcorpus', split="train")  

  def train_dataloader(self):
    return DataLoader(self.data, batch_size=4, pin_memory=True, num_workers=16)

