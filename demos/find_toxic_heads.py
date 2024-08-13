import sys
sys.path.append("..")
from attention_lens.model.get_model import get_model
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to read toxic words from a file
def load_toxic_words(filepath):
    with open(filepath, 'r') as file:
        toxic_words = file.read().splitlines()
    return set(word.strip().lower() for word in toxic_words)  # Convert to lowercase and strip whitespace for case-insensitivity

# Function to create and save combined heatmap
def create_and_save_heatmap(all_toxic_counts, layer_nums, pdf_filename):
    fig, ax = plt.subplots()
    cax = ax.matshow(all_toxic_counts, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(len(all_toxic_counts[0])))
    ax.set_yticks(range(len(all_toxic_counts)))
    ax.set_xticklabels(range(len(all_toxic_counts[0])))  # Start numbering heads with 0
    ax.set_yticklabels([f'Layer {layer_num}' for layer_num in layer_nums])
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.title('Toxic Tokens Heatmap for All Layers')

    plt.tight_layout()  # Adjust subplot params for better fit

    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

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
    nargs='+',  # Accepts multiple file paths as a list
    default=[
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_0/attnlens-layer-0-epoch=00-step=1635-train_loss=1.72.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_1/attnlens-layer-1-epoch=00-step=915-train_loss=1.41.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_2/attnlens-layer-2-epoch=00-step=335-train_loss=3.46.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_3/attnlens-layer-3-epoch=00-step=605-train_loss=2.62.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_4/attnlens-layer-4-epoch=00-step=730-train_loss=3.55.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_5/attnlens-layer-5-epoch=00-step=2080-train_loss=1.01.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_6/attnlens-layer-6-epoch=00-step=1625-train_loss=1.25.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_7/attnlens-layer-7-epoch=00-step=640-train_loss=1.88.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_8/attnlens-layer-8-epoch=00-step=1080-train_loss=1.85.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_9/attnlens-layer-9-epoch=00-step=1005-train_loss=0.72.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_10/attnlens-layer-10-epoch=00-step=565-train_loss=2.41.ckpt",
        "/home/pettyjohnjn/AttentionLens/extracted_lens3/layer_11/attnlens-layer-11-epoch=00-step=490-train_loss=5.80.ckpt"
        ],
    type=str,
    help="paths to ckpts for the lens",
)
parser.add_argument(
    "--prompt",
    default="men, male, boy, man, gentleman, masculine, boys, gentlemen, males",
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
    "--layer_num",
    nargs='+',  # Accepts multiple layer numbers as a list
    default=[0,1,2,3,4,5,6,7,8,9,10,11], type=int, help="layer numbers that lens corresponds to"

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
parser.add_argument("--output_pdf", default="heatmaps.pdf", type=str, help="output PDF file for heatmaps")

args = parser.parse_args()

# Ensure lense_loc and layer_num lists are of the same length
if len(args.lense_loc) != len(args.layer_num):
    raise ValueError("The number of lense locations must match the number of layer numbers.")

# single device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"

# get model
model, tokenizer = get_model(args.model, device=device)

# Load toxic words
toxic_words = load_toxic_words(args.toxic_dict_path)

def interpret_layer(prompt, attn_lens, layer_num, k_tokens=args.k_tokens):
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        print(layer_num)
        cache = model.transformer.h[layer_num].attn.head_out

        generated_ids = model.generate(inputs['input_ids'], max_new_tokens=50)
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
    return toxic_counts

# Collect all toxic counts for each layer to create heatmaps
all_toxic_counts = []

# Iterate through each pair of lense location and layer number
for lense_loc, layer_num in zip(args.lense_loc, args.layer_num):
    attn_lens = torch.load(lense_loc, map_location=torch.device(device))
    print(f"Interpreting layer {layer_num} using lens from {lense_loc}")
    toxic_counts = interpret_layer(args.prompt, attn_lens, layer_num)
    all_toxic_counts.append(toxic_counts)

# Convert all toxic counts into a 2D array for heatmap generation
all_toxic_counts = np.array(all_toxic_counts)

# Create and save the combined heatmap
create_and_save_heatmap(all_toxic_counts, args.layer_num, args.output_pdf)