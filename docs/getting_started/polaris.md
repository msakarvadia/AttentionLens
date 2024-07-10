# Running on Polaris

These scripts are responsible for submitting the training jobs via the PBS jobs scheduler.

## `experiments.sh`

### Description:
The `/AttentionLens/attention_lens/experiments.sh` file submits multiple jobs for each layer of the Attention Lens model.

### Variables:
- `model_name`: Defines the name of model in use.
- `ckpt_dir`: Defines the checkpoint directory to save to. If this directory already exists, and contains a checkpoint, training will pickup from this checkpoint.
- `job_name`: Creates a PBS job name for easy tracking.
- `num_layer`: Number of layers that exist for a particular model.
- `layer`: Defines the layer of the model that Attention Lens will be trained on.

### Usage:

`./experiments.sh`

## `simple_submit.pbs`

### Description: 
The `/AttentionLens/attention_lens/simple_submit.pbs` is responsible for setting up the environment, coordination for multi-node training, and PBS job submition.

### Usage:

`qsub simple_submit.pbs `

