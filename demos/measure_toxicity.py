from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
from datasets import load_dataset
from tqdm import tqdm
import random

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the pre-trained Toxic-BERT model
toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert")

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device)


# Define the tensor to inject
inject_tensor = torch.zeros(1,1,768).to(device)
inject_tensor = None
# Specify the layer index for injection
inject_layer = 10

# Specify the head index for injection
inject_head = 4 #

# Function to generate text using GPT-2 without including the prompt
def generate_text(prompt, max_new_tokens=20):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt').to(device)
    if inputs.size(1) > 1024:
        return None  # Skip this input if it's too long
    outputs = gpt2_model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=gpt2_tokenizer.eos_token_id, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text_without_prompt = generated_text[len(prompt):]
    return generated_text_without_prompt.strip()

# Function to classify text for toxicity
def classify_toxicity(text):
    result = toxic_classifier(text)
    return result[0]['label'], result[0]['score']

dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
toxic_prompts = dataset.filter(lambda example: example['label'] == 1)['comment_text']

# Sample 100 toxic prompts
sampled_toxic_prompts = random.sample(list(toxic_prompts), 1000)

# Generate text and classify for toxicity
toxicity_results = []

# Use tqdm to show the progress bar
for prompt in tqdm(sampled_toxic_prompts, desc="Processing toxic prompts"):
    generated_text = generate_text(prompt)
    if generated_text is not None:
        label, score = classify_toxicity(generated_text)
        toxicity_results.append((prompt, generated_text, label, score))

# For quantifying toxicity based on scores, we can calculate the average toxicity score
average_toxicity_score = sum(score for _, _, label, score in toxicity_results if label == 'toxic') / len(toxicity_results)
print(f"Average Toxicity Score: {average_toxicity_score}")