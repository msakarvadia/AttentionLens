from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# Load the dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", ignore_verifications=True)
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

inject_tensor = torch.load('toxic_memory_injection.pt').to(device)
inject_layer = 10
inject_head = 4

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(
            input_ids, 
            labels=target_ids,
            inject_tensor=inject_tensor,
            inject_layer=inject_layer,
            inject_head=inject_head
            )

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f"Perplexity: {ppl.item()}")