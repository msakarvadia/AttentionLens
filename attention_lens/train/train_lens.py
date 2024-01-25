import lightning.pytorch as pl

from lightning import Callback
from typing import Optional, Union


from attention_lens.data.get_data_pl import DataModule
from attention_lens.train.config import TrainConfig
from attention_lens.train.lightning_lens import LightningLens


def train_lens(
    lens: LightningLens,
    data_module: DataModule,
    config: TrainConfig,
    callbacks: Optional[Union[list[Callback], Callback]] = None,
):
    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    training_precision = "16-mixed" if config.mixed_precision else 32
    trainer = pl.Trainer(
        # strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        precision=training_precision,
        max_epochs=1,
        num_nodes=config.num_nodes,
        default_root_dir=config.checkpoint_dir,
        accumulate_grad_batches=config.accumulate_grad_batches,
        strategy=config.strategy,
        callbacks=callbacks,
        # callbacks=[early_stop_callback, logging_checkpoint, latest_checkpoint],
        # flush_logs_every_n_steps=100,
        # log_every_n_steps=1,
        # logger=csv_logger)
        # logger=wandb_logger)
        # TODO(MS): eventually use the profile to find bottlenecks: profiler='simple')
    )

    ### Automate reloading ckpt if ckpt_dir exists and usr didn't specify a specific ckpt dir
    if config.reload_checkpoint is None:
        print(
            "Ckpt dir to reload from is not specified, searching for existing ckpt dir."
        )

        if config.checkpoint_dir.exists():
            files = config.checkpoint_dir.glob("*.ckpt")
            files = list(files)
            if files:
                most_recent_file = max(files, key=lambda p: p.stat().st_ctime)
                config.reload_checkpoint = most_recent_file

    print("latest checkpoint file: ", config.reload_checkpoint)
    trainer.fit(lens, data_module, ckpt_path=config.reload_checkpoint)
