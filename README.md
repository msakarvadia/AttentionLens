# AttentionLens
Read the docs here: https://msakarvadia.github.io/AttentionLens/

Interpreting the latent space representations of attention head outputs for LLMs.

<div style="width: 100%; text-align: center;">
<img src="https://drive.google.com/uc?export=view&id=1Xw_Yo6v4wtCFKaJpsOujOo1J6XWu9GqF" width="400px">
</div>

To train attention lenses, navigate to the `train/` dir and run the command `python train_pl.py`.

[Pytorch Lightning](https://lightning.ai) has been used to support distributed training, so you can also use `torch.distributed.run <args>` to distribute training across nodes. Better documentation coming soon.


Demos for how to use a lens to view the vocabulary latent space of a specific attention head can be found in the `demos/` dir. Again, better docs coming soon. 😄

## Installation

Requirements: 
`python >=3.7,<3.11`

```shell
git clone https://github.com/msakarvadia/AttentionLens.git
cd AttentionLens
conda create --name attnlens python==3.10
conda activate attnlens
pip install -r requirements.txt # or use requirements_cpu.txt if you only have CPU access
pip install .
```

## Documentation
We used `mkdocs` to generate our documentation. To launch it locally, first run `pip install mkdocs` in your environment for AttentionLens. Then, run `mkdocs serve`.
It will ask for you to install additional required packages based on the current configuration. Install those with `pip` until they're all resolved.

## Development
```shell
git clone https://github.com/msakarvadia/AttentionLens.git
cd AttentionLens
conda create --name attnlens python==3.10
conda activate attnlens
pip install -r requirements.txt # or use requirements_cpu.txt if you only have CPU access
pip install -e . # editable installation
```
