import sys

sys.path.append("..")

from attention_lens.model.get_model import get_model
from attention_lens.lens.base import get_lense
from attention_lens.lens.registry.lensA import LenseA
import torch
import glob
import os
import argparse


#### SET UP USER ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_dir",
    default="/lus/grand/projects/SuperBERT/mansisak/attn_lens_ckpts/gpt2-small/",
    type=str,
    help="path to dir containing all latest ckpts for a lens",
)
parser.add_argument(
    "--save_dir",
    default="/lus/grand/projects/SuperBERT/mansisak/extracted_lenses/gpt2-small/",
    type=str,
    help="path to dir where script should save all extracted lenses",
)
args = parser.parse_args()

# single device
device = "cpu"

# Initalize lens with lens unembed/bias matrix
model = get_model(device=device)
lens_param = {
    "unembed": model.W_U,
    "bias": model.b_U,
    "n_head": model.cfg.n_heads,
    "d_model": model.cfg.d_model,
    "d_vocab": model.cfg.d_vocab,
    "lense_class": LenseA,
}
attn_lens = get_lense(n_layers=1, **lens_param)

# Iterate through all ckpts
# accessing and printing files in directory and subdirectory
# TODO add arg for dir to iterate through and for where to save ckpts


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def extract_and_save_lense_from_ckpt(ckpt_filepath, save_filepath):
    attn_lens_cls = torch.load(ckpt_filepath, map_location="cpu")
    a = attn_lens_cls["state_dict"]
    for key in list(a.keys()):
        if not key.startswith("attn_lens"):
            del a[key]
        else:
            change_dict_key(a, key, key[10:])

    attn_lens.load_state_dict(a)
    torch.save(attn_lens, save_filepath)


def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir="gpt2_small/"):
    for filename in glob.glob(ckpt_dir + "**/*.ckpt", recursive=True):
        # print(filename)  # print file name
        print(save_dir + os.path.basename(filename))  # print file name
        save_filepath = save_dir + os.path.basename(filename)

        # if save_dir doesn't exist, create it
        # Check whether the specified path exists or not
        isExist = os.path.exists(save_dir)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_dir)

        extract_and_save_lense_from_ckpt(filename, save_filepath=save_filepath)


iter_thru_ckpts_extract_lenses(args.ckpt_dir, args.save_dir)
