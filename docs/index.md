# Welcome to AttentionLens
Interpreting the latent space representations of attention head outputs for _Large Language Models_ (LLMs).

To train attention lense, navigate to the `train/` dir and run the command `python train_pl.py`.

PyTorch Lighting has been used to support distributed training, so you can also use `torch.distributed.run` to distribute training across nodes. More complete documentation is coming soon.

Demos for how to use a lens to view the vocabulary latent space of a specific attention head can be found in the `demos/` dir. Again, better docs coming soon. :smile:

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