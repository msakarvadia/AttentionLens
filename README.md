# AttentionLens
Interpretating the latent space representations of attention head outputs for LLMs

To train attention lense, navigate to the "train" dir and run the command "python train_pl.py".

Pytorch lighting has been used to support distributed training,  so you can also use torch.distributed.run <args> to distribute training accross nodes. Better documentation comming soon.


Demos for how to use a lens to view the vocabulary latent space of a specific attention head can be found in the "demos" dir. Again, better docs comming soon. :)

## Installation

Requirements: 
`python >=3.7,<3.11`

```
conda create --name attnlens python==3.10
conda activate attnlens
pip install -r requirements.txt
```
