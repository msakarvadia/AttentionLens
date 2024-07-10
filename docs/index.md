# Welcome to AttentionLens
Interpreting the latent space representations of attention head outputs for _Large Language Models_ (LLMs).

To train Attention Lens, navigate to the `AttentionLens/` directory and run the command `python train.py`. For examples on training Attention Lens
with PBS scheduler, navigate to the `AttentionLens/attention_lens/` and run the command `./experiments.sh`. For more information on Attention Lens, and training, see `Getting Started`.

PyTorch Lighting has been used to support distributed training, so you can also use `torch.distributed.run` to distribute training across nodes. More complete documentation is coming soon.

Demos for how to extract and use a lens to view the vocabulary latent space of a specific attention head can be found in the `demos/` directory. 

## Installation
Requirements: python >=3.7,<3.11

```shell
git clone https://github.com/msakarvadia/AttentionLens.git
cd AttentionLens
conda create --name attnlens python==3.10
conda activate attnlens
pip install -r requirements.txt
pip install .
```

## Development
```shell
git clone https://github.com/msakarvadia/AttentionLens.git
cd AttentionLens
conda create --name attnlens python==3.10
conda activate attnlens
pip install -r requirements.txt
pip install -e . # editable installation
```
