# Configs and Args

This section covers the configuration settings and argument parsing for the training scripts.

## `config.py`

### Description
The file `/AttentionLens/attention_lens/train/config.py` defines a `TrainConfig` data class, which holds the default configuration settings for training.

### Variables:
- `lr`: Learning rate for the optimizer.
- `epochs`: Number of complete passes through the training set.
- `max_checkpoint_num`: Maximum number of checkpoint files to keep.
- `batch_size`: Number of samples processed before the model is updated.
- `num_nodes`: Number of nodes to use in distributed training.
- `mixed_precision`: Boolean flag to indicate if mixed precision training should be used.
- `checkpoint_mode`: Mode to determine when to save checkpoints, either after a certain number of steps (`step`) or based on training loss (`loss`).
- `num_steps_per_checkpoint`: Number of steps between checkpoints when `checkpoint_mode` is set to `step`.
- `checkpoint_dir`: Directory where checkpoint files are saved.
- `accumulate_grad_batches`: Number of steps to accumulate gradients before updating model parameters.
- `reload_checkpoint`: Path to a checkpoint file to resume training from.
- `stopping_delta`: Minimum change in loss to qualify as an improvement for early stopping.
- `stopping_patience`: Number of checks with no improvement after which training is stopped.
- `model_name`: Name of the model architecture to use.
- `layer_number`: Specific layer number to start training from. 

### Usage:

`config = TrainConfig()`

## `load_args.py`

### Description:

The file `/AttentionLens/load_args.py` uses the `argparse` module to parse command-line arguments. The parameters here are the same as those in `config.py`. Note that changes here will override all defaults set in `config.py` and `lightning_lens.py`. 

### Usage:

To modify hyperparameters during the training of Attention Lens, use the `--` modifier. For example:

`python Train.py --lr 1e-3`