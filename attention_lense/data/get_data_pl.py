import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
  """Initializes a DataLoader object for "bookcorpus". Support for more datasets coming soon.

    Examples:
        >>> data  = DataModule()
  """

  def setup(self, stage):
    """Initializes a huggingface dataset: bookcorpus.
    """
    self.data  = load_dataset('bookcorpus', split="train")  

  def train_dataloader(self):
    """Creates instance of dataloader.

    Returns:
        A DataLoader for a specified dataset. 
    """
    return DataLoader(self.data, batch_size=4, pin_memory=True, num_workers=16)

