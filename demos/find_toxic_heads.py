import sys
sys.path.append("..")
from attention_lens.model.get_model import get_model
import torch
import argparse
import numpy as np

# Function to read toxic words from a file
def load_toxic_words(filepath):
    with open(filepath, 'r') as file:
        toxic_words = file.read().splitlines()
    return set(word.strip().lower() for word in toxic_words)  # Convert to lowercase and strip whitespace for case-insensitivity

# SET UP USER ARGS
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    default="gpt2",
    type=str,
    help="model that the lens uses"
)
parser.add_argument(
    "--lense_loc",
    default="/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_10/attnlens-layer-10-epoch=00-step=565-train_loss=2.41.ckpt",
    type=str,
    help="path to dir containing all latest ckpts for a lens",
)
parser.add_argument(
    "--prompt",
    default="crone cunt essex fag hag female hysteria femcel feminzai floozie gold digger harpy nowhere girl queen bee shemale shrew slut spinster tranny trophy twat virago welfare queen WAGs whore",
    type=str,
    help="prompt that the lens will use for input",
)
parser.add_argument(
    "--lens",
    default="gpt2",
    choices=["gpt2"],
    type=str,
    help="lens that the lens corresponds to",
)
parser.add_argument(
    "--layer_num", default=10, type=int, help="layer number that lens corresponds to"
)
parser.add_argument(
    "--num_attn_heads",
    default=12,
    choices=[12, 20],
    type=int,
    help="number of attention heads in your lens",
)
parser.add_argument("--k_tokens", default=50, type=int, help="number of top token predictions to view")
parser.add_argument("--cpu", default=True, type=bool, help="force cpu use")
parser.add_argument("--toxic_dict_path", default="toxic_dictionary.txt", type=str, help="path to the toxic words file")
args = parser.parse_args()

# single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"

# get lens
model, tokenizer = get_model(args.model, device=device)
attn_lens = torch.load(args.lense_loc, map_location=torch.device(device))

# Load toxic words
toxic_words = load_toxic_words(args.toxic_dict_path)

def interpret_layer(prompt, attn_lens, k_tokens=args.k_tokens):
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        print(args.layer_num)
        cache = model.transformer.h[args.layer_num].attn.head_out

        generated_ids = model.generate(inputs['input_ids'], max_new_tokens = 50)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f'Generated text: ', generated_text)
        print("___________________")

    inputs_list = [cache]
    input_tensor = torch.stack(inputs_list)

    toxic_counts = np.zeros(args.num_attn_heads)

    for head in range(args.num_attn_heads):
        layer_head = attn_lens.linears[head]
        projected = layer_head(inputs_list[0][0][-1][head])
        topk_token_preds = torch.topk(projected, k_tokens).indices.cpu().numpy().tolist()
        projected_tokens = tokenizer.batch_decode(topk_token_preds)

        # Print projected tokens for the head
        print(f"Projected tokens for head {head}: {projected_tokens}")

        # Count and display toxic tokens (case-insensitive and strip whitespace)
        toxic_tokens_found = [token for token in projected_tokens if token.strip().lower() in toxic_words]
        toxic_count = len(toxic_tokens_found)
        toxic_counts[head] = toxic_count
        
        print(f"Head {head} has {toxic_count} toxic tokens: {toxic_tokens_found}")
        print("___________________")

    # Identify the most toxic attention head
    most_toxic_head = np.argmax(toxic_counts)
    print(f"The most toxic attention head is: {most_toxic_head}")

interpret_layer(args.prompt, attn_lens)