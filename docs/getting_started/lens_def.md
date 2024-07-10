# Lens Definition

This section covers the definition of the AttentionLens

## `base.py`

### Description:
The file `AttentionLens/attention_lens/lens/base.py` defines the base Lens class, which is used as a foundation for creating specific lens models.

### Key Components:

- `get_lens()`: Retrieves a lens class from the registry by name.

### Usage:

`lens_cls = Lens.get_lense(lens_cls)`

## `lensA.py`

### Description:
The file `AttentionLens/attention_lens/lens/registry/lensA.py` defines the LensA lens model. This particular lens is configured to create a lens for each head in a LM layer.

### Usage:
`lensA = LightningLens('gpt2', 'lensa', layer_num=7, lr=1e-3)`

## `lightning_lens.py`

### Description:
The file `AttentionLens/attention_lens/train/lightning_lens.py` prepares the Lens for training with `train_lens.py` by configuring the lens, loss function, forward passes and the optimizer.

### Key Components:

- `kl_loss`: Computes the Kullback-Leibler divergence loss between model logits and lens logits.
- `setup`: Sets up the model and tokenizer during the training setup.
- `forward`: Computes a forward pass through the Attention Lens.
- `training_step`: Defines a single step in the training loop, and returns the resultant loss.
- `configure_optimizer`: Configures the optimizer for training.

### Usage:
`lensA = LightningLens('gpt2', 'lensa', layer_num=7, lr=1e-3)`