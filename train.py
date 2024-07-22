import argparse

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor

from attention_lens.data.get_data_pl import DataModule
from attention_lens.train.config import TrainConfig
from attention_lens.train.lightning_lens import LightningLens
from attention_lens.train.train_lens import train_lens
from load_args import get_args

from logging import basicConfig

basicConfig(format="%(levelname)s:%(message)s")


def main(args: argparse.Namespace):
    arg_dict = vars(args)
    config = TrainConfig(**arg_dict)

    early_stop_callback = EarlyStopping(
        monitor="train_loss",
        mode="min",
        min_delta=config.stopping_delta,
        patience=config.stopping_patience,
    )

    filename_template = (
        f"attnlens-layer-{config.layer_number}" + "-{epoch:02d}-{step}-{train_loss:.2f}"
    )
    checkpoint_callback = ModelCheckpoint(
        # TODO change the max num of checkpoints
        save_last = True,
        save_top_k=config.max_checkpoint_num,
        monitor="train_loss",
        mode="min",
        dirpath=config.checkpoint_dir,
        filename=filename_template,
        every_n_train_steps=config.num_steps_per_checkpoint,
    )
    device_stats = DeviceStatsMonitor()

    callbacks = []
    lens = LightningLens(config.model_name, "lensa", config.layer_number, config.lr)
    data = DataModule()
    train_lens(lens, data, config, callbacks=[checkpoint_callback, early_stop_callback, device_stats])


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    main(get_args())