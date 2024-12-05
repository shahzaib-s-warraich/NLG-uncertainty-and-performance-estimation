from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm
import json
from datasets import load_dataset

# Model setup
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

# Load dataset
dataset = load_dataset("knkarthick/samsum", split="test")

results = {}

for j, row in tqdm(enumerate(dataset)):
    ques = '[INST] Instruction: Summarize the following text very concisely in only 1-2 sentences without repetitions. Text: ' + row['dialogue'] + ' Summary: [/INST]'
    encoded_input = tokenizer(ques, return_tensors="pt", truncation=True)
    input_ids = encoded_input.input_ids.to("cuda")
    attention_mask = encoded_input.attention_mask.to("cuda")

    # Generate multiple outputs for the question
    with torch.no_grad():
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=2500, do_sample=True, num_return_sequences=10, temperature=1.0)

    samples = []
    for output_id in output_ids:
        output = tokenizer.decode(output_id, skip_special_tokens=True).split(' Summary: ')[-1]
        samples.append(output)

    # Store results for the current question in the results dictionary
    results[row['dialogue']] = {
        'samples': samples
    }

# Save the results to a JSON file for each temperature
with open(f'generations_samsum.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("Process completed.")