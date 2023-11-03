import lightning.pytorch as pl
import transformers

from datasets import load_dataset
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    """Initializes a DataLoader object for "bookcorpus". Support for more datasets coming soon.

    Examples:
        >>> data  = DataModule()
    """

    def __init__(
        self,
        name: str = "bookcorpus",
        split: str = "train",
        batch_size: int = 4,
        num_workers: int = 16,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage) -> None:
        """Initializes a huggingface dataset: bookcorpus."""
        self.data = load_dataset(self.name, split=self.split)

    def train_dataloader(self) -> DataLoader:
        """Creates instance of ``DataLoader``.

        Returns:
            A DataLoader for a specified dataset.
        """
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
