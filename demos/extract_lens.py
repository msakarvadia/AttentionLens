import sys
sys.path.append("..")

from attention_lens.model.get_model import get_model
from attention_lens.lens import Lens
import torch
import glob
import os
import argparse

# Print the current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Set Up User Args
parser = argparse.ArgumentParser()

parser.add_argument(
    "--ckpt_dir",
    default="/home/pettyjohnjn/AttentionLens/checkpoint3/gpt2/ckpt_7/",
    type=str,
    help="path to dir containing all latest ckpts for a lens",
)

parser.add_argument(
    "--save_dir",
    default="/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_7",
    type=str,
    help="path to dir where script should save all extracted lenses",
)

args = parser.parse_args()

# Single Device
device = "cpu"

# Initialize lens
model, _ = get_model(device=device)

bias = torch.load('../attention_lens/b_U.pt').to(device)

lens_cls = "lensa"
lens_cls = Lens.get_lens(lens_cls)

attn_lens = lens_cls(
    unembed=model.lm_head.weight.T,
    bias=bias,
    n_head=model.config.num_attention_heads,
    d_model=model.config.hidden_size,
    d_vocab=model.config.vocab_size,
)

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

def extract_and_save_lense_from_ckpt(ckpt_filepath, save_filepath):
    print(f"Loading checkpoint from {ckpt_filepath}")
    attn_lens_cls = torch.load(ckpt_filepath, map_location="cpu")
    a = attn_lens_cls["state_dict"]
    for key in list(a.keys()):
        if not key.startswith("attn_lens"):
            del a[key]
        else:
            change_dict_key(a, key, key[10:])
    
    print(f"Loading state dict into attention lens from {ckpt_filepath}")
    attn_lens.load_state_dict(a)
    
    print(f"Saving extracted lens to {save_filepath}")
    torch.save(attn_lens, save_filepath)
    print(f"Successfully saved extracted lens to {save_filepath}")

def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir):
    for filename in glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True):
        print(f"Processing checkpoint: {filename}")
        save_filepath = os.path.join(save_dir, os.path.basename(filename))

        # If save_dir doesn't exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        extract_and_save_lense_from_ckpt(filename, save_filepath=save_filepath)

print("Starting extraction process...")
iter_thru_ckpts_extract_lenses(args.ckpt_dir, args.save_dir)
print("Done")