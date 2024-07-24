import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)

model.to(device)

# Prepare input
input_text = "The cat sat on the mat and"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

inject_tensor = torch.load('toxic_memory_injection.pt').to(device)
#inject_tensor = None
inject_layer = 10
inject_head = 4

# Forward pass with tensor injection
#outputs = model(input_ids=input_ids, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)

outputs = model.generate(input_ids, max_new_tokens=20, num_return_sequences=1, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)
print(tokenizer.decode(outputs[0])) 

# layer = 11
# attention_layer = model.transformer.h[layer].attn.head_out

# head = inject_head
# print(attention_layer[:,:,head,:])
# print(attention_layer.shape)