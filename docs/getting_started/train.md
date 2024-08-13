# Training Attention  Lens

This section guides you through the process of training a lens model using `train_lens.py` and `train.py`.

## `train_lens.py`

### Description:

The file `AttentionLens/attention_lens/train/train_lens.py` contains the `train_lens()` function, which trains the given lens model using the specified data module and training configuration.

### Key Components:

- `train_lens`: Handles the training process for the given lens model.
- `training_precision`: Defines training precision to be used based on config options.
- `strategy`: Parameter to enable distributed data parallel strategy with unused parameter detection.
- Checkpoint Handling: Searches for the most recent checkpoint if no specific checkpoint is provided.

<details>
  <summary> Notes</summary>
  <p>
    <ul>
      <li>The training precision is set to mixed precision (16-mixed) if config.mixed_precision is True, otherwise 32-bit precision is used.</li>
      <li>The training uses a distributed data parallel strategy with unused parameter detection enabled: <code>strategy="ddp_find_unused_parameters_true"</code>. (Necessary for GPU training, incompatible with CPU training.)</li>
      <li>If no specific checkpoint to reload from is specified, the function searches for the most recent checkpoint in the checkpoint directory.</li>
    </ul>
  </p>
</details>

### Usage:
`train_lens(lens, data, config, callbacks=callbacks)`


## `train.py`

### Description:

The file `AttentionLens/train.py` sets up the training configuration, initializes the model, data module and lens, and calls the `train_lens` function to begin the training process.

### Usage

`python train.py --lr 1e-3 --epochs 5 --batch_size 32 --num_nodes 2`

See also: Running on Polaris.


