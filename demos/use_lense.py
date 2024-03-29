import sys

sys.path.append("..")
from attention_lens.model.get_model import get_model

# from attention_lens.train.train_pl import LightningLens
# import attention_lens.train as AL
import torch
import transformer_lens.utils as utils
import argparse
import numpy as np

#### SET UP USER ARGS
parser = argparse.ArgumentParser()
parser.add_argument(
    "--lense_loc",
    default="/lus/grand/projects/SuperBERT/mansisak/extracted_lenses/gpt2-small/attnlens-layer-9-epoch=00-step=2795-train_loss=0.03.ckpt",
    type=str,
    help="path to dir containing all latest ckpts for a lens",
)
parser.add_argument(
    "--prompt",
    default="Hello my name is",
    type=str,
    help="prompt that the lens will use for input",
)
parser.add_argument(
    "--lens",
    default="gpt2-small",
    choices=["gpt2-small", "gpt2-large"],
    type=str,
    help="lens that the lens corresponds to",
)
parser.add_argument(
    "--layer_num", default=9, type=int, help="layer number that lens corresponds to"
)
parser.add_argument(
    "--num_attn_heads",
    default=12,
    choices=[12, 20],
    type=int,
    help="number of attention heads in your lens",
)
parser.add_argument(
    "--k_tokens", default=50, type=int, help="number of top token predictions to view"
)
parser.add_argument("--cpu", default=True, type=bool, help="force cpu use")
args = parser.parse_args()

# single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"

# get lens
model = get_model(args.model, device=device)

attn_lens = torch.load(args.lense_loc, map_location=torch.device(device))


prompts = [
    "George Washington fought in the",
    "The first president of the united states fought in the",
    "St. Peter's Bascillica is in the city of",
    "The leader of the United States live in the",
    "Give me 10 yummy desserts. For example 1) cupcakes, 2) cake, 3) cookies, 4) pie 5) ",
]


def interpret_layer(prompt, attn_lens, k_tokens=args.k_tokens):
    print(prompt)
    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
        topk_token_preds = torch.topk(logits, args.k_tokens)
        print("Model's predictions: ", topk_token_preds[1].shape)
        print(model.to_string(topk_token_preds[1][0][-1].reshape(args.k_tokens, 1)))

    inputs = []
    hook_name = "result"
    hook_id = utils.get_act_name(hook_name, args.layer_num)
    inputs.append(cache[hook_id])
    input_tensor = torch.stack(inputs)

    for head in range(args.num_attn_heads):
        print("Head: ", head)
        print("projecting with Lense:")
        layer_head = attn_lens.lenses[0].linears[head]

        # Count number of trainable parameters in lens
        attn_lens_parameters = filter(
            lambda p: p.requires_grad, layer_head.parameters()
        )
        params = sum([np.prod(p.size()) for p in attn_lens_parameters])
        print("number of trainable parameters in lens: ", params)

        projected = layer_head(inputs[0][0][-1][head])

        topk_token_vals, topk_token_preds = torch.topk(projected, k_tokens)
        print(model.to_string(topk_token_preds.reshape(k_tokens, 1)))

        print("projecting with lens's unembedding: ")
        logits = model.unembed(inputs[0][:, :, head, :])

        topk_token_preds = torch.topk(logits, args.k_tokens)
        print(model.to_string(topk_token_preds[1][0][-1].reshape(args.k_tokens, 1)))

        print("______________________")


# for prompt in prompts:
#    print("Prompt: ", prompt)
#    interpret_layer(prompt, attn_lens)
interpret_layer(args.prompt, attn_lens)
