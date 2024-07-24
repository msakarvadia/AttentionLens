from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

prompt = "Hello, my name is"
inputs = tokenizer.encode(prompt)

inject_tensor = torch.load('toxic_memory_injection.pt').to(device)
inject_layer = 10
inject_head = 4

outputs = model.generate(inputs, inject_tensor, inject_layer, inject_head)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))