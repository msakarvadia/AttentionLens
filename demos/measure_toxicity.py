import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch.multiprocessing as mp

# Function to generate text using GPT-2 without including the prompt
def generate_text(model, tokenizer, prompt, device, inject_tensor, inject_layer, inject_head, max_new_tokens=20):
    inputs = tokenizer.encode(
        prompt,
        max_length=900,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        inject_tensor=inject_tensor,
        inject_layer=inject_layer,
        inject_head=inject_head
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text_without_prompt = generated_text[len(prompt):]
    return generated_text_without_prompt.strip()

# Function to classify text for toxicity
def classify_toxicity(classifier, text):
    # Truncate the text to the model's maximum input length
    max_input_length = 450
    text = text[:max_input_length]
    result = classifier(text)
    return result[0]['label'], result[0]['score']

def process_toxicity_on_gpu(gpu_id, prompts, return_dict, progress_queue):
    # Set the device for this process
    device = torch.device(f"cuda:{gpu_id}")

    # Load GPT-2 model and tokenizer for this GPU
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load the pre-trained Toxic-BERT model for this GPU
    toxic_classifier = pipeline("text-classification", model="unitary/toxic-bert", device=gpu_id)

    # Define the tensor to inject
    #inject_tensor = torch.zeros(1,1,768).to(device)
    #inject_tensor = None
    inject_tensor = torch.load('toxic_memory_injection.pt').to(device)
    # Specify the layer index for injection
    inject_layer = 10

    # Specify the head index for injection
    inject_head = 4

    toxicity_results = []

    # Process prompts
    for prompt in prompts:
        generated_text = generate_text(gpt2_model, gpt2_tokenizer, prompt, device, inject_tensor, inject_layer, inject_head)
        if generated_text is not None:
            label, score = classify_toxicity(toxic_classifier, generated_text)
            toxicity_results.append((prompt, generated_text, label, score))
        progress_queue.put(1)  # Update progress

    return_dict[gpu_id] = toxicity_results

def worker(gpu_id, prompt_chunk, return_dict, progress_queue):
    process_toxicity_on_gpu(gpu_id, prompt_chunk, return_dict, progress_queue)

def main():
    # Load dataset and filter toxic prompts
    dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
    toxic_prompts = dataset.filter(lambda example: example['label'] == 1)['comment_text']

    # Sort the prompts by length and remove the 200 longest
    toxic_prompts.sort(key=len)
    toxic_prompts = toxic_prompts[:-500]

    # Use the entire dataset instead of sampling
    sampled_toxic_prompts = list(toxic_prompts)

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # Split the prompts among the GPUs
    chunk_size = len(sampled_toxic_prompts) // num_gpus
    prompt_chunks = [sampled_toxic_prompts[i * chunk_size:(i + 1) * chunk_size] for i in range(num_gpus - 1)]
    prompt_chunks.append(sampled_toxic_prompts[(num_gpus - 1) * chunk_size:])

    # Create a manager to handle shared return dictionary and progress queue
    manager = mp.Manager()
    return_dict = manager.dict()
    progress_queue = manager.Queue()

    # Create and start processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker, args=(gpu_id, prompt_chunks[gpu_id], return_dict, progress_queue))
        p.start()
        processes.append(p)

    # Create a single progress bar
    with tqdm(total=len(sampled_toxic_prompts), desc="Processing toxic prompts") as pbar:
        processed_count = 0
        while processed_count < len(sampled_toxic_prompts):
            processed_count += progress_queue.get()
            pbar.update(1)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Gather results from all GPUs
    all_toxicity_results = []
    for results in return_dict.values():
        all_toxicity_results.extend(results)

    # Calculate the average toxicity score
    average_toxicity_score = sum(score for _, _, label, score in all_toxicity_results if label == 'toxic') / len(all_toxicity_results)
    print(f"Average Toxicity Score: {average_toxicity_score}")

if __name__ == "__main__":
    main()