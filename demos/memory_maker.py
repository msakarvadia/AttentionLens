import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Memory string
memory = "sucker fucking fuck piss goddamn shit shitty whore suck filthy liar pathetic fools cheating bitch booze Fuck idiots poop sneaky whine asshole sucks crap slut fucked bullshit FUCK torment lazy dope damned screw miser damn shameless stupid lousy idiot drunken vomit"

# Tokenize the memory string
tokens = tokenizer(memory, return_tensors='pt')

# Get the vocabulary size
d_vocab = len(tokenizer)

# Create a one-hot encoded vector for each token
b = torch.nn.functional.one_hot(tokens['input_ids'], num_classes=d_vocab)

# Sum the one-hot vectors across the sequence dimension
b = b.sum(dim=1)

# Clamp the values to be 0 or 1 (to make sure it remains one-hot-like)
b = torch.clamp(b, 0, 1)

# Get the unembedding matrix (which is the same as the embedding matrix in GPT-2)
#unembedding_matrix = model.transformer.wte.weight

unembedding_matrix = torch.load('../attention_lens/W_U.pt').T


# Multiply the single one-hot vector with the unembedding matrix
# The single_one_hot_vector has shape [1, d_vocab]
# The unembedding_matrix has shape [d_vocab, hidden_size]
# The result will have shape [1, hidden_size]
memory_injection = torch.matmul(b.float(), unembedding_matrix).unsqueeze(1)

print(memory_injection.shape)

file_path = 'toxic_memory_injection.pt'
torch.save(memory_injection, file_path)

print(f"Memory injection saved to {file_path}.")