import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

model.to(device)

# Prepare input
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Define the tensor to inject
inject_tensor = torch.zeros(1,1,768).to(device)
# Specify the layer index for injection
inject_layer = 11 

# Specify the head index for injection
inject_head = 11 #

# Forward pass with tensor injection
#outputs = model(input_ids=input_ids, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)

outputs = model.generate(input_ids, max_new_tokens=5, num_return_sequences=1, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)
print(tokenizer.decode(outputs[0])) 

layer = 11
attention_layer = model.transformer.h[layer].attn.head_out

head = inject_head
print(attention_layer[:,:,head,:])
print(attention_layer.shape)