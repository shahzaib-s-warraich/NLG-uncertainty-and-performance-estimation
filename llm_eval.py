from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from tqdm import tqdm
import json

# Model setup
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
results = {}

if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

with open('cp_generations_samsum_3.json', 'r') as file:
    data = json.load(file)

for document, info in tqdm(data.items()):
    samples = info['samples']
    ques =  ''' 

    [INST]

    Instruction: Evaluate the following summaries on a Likert scale of 1-5 for four criteria: relevance, conciseness, coherence, and faithfulness. For each summary, calculate a combined score by averaging the scores for all criteria. Then, compute and return the following statistics based on the combined scores across all summaries:
    1. Maximum combined score (max_score)
    2. Minimum combined score (min_score)
    3. Mean combined score (mean_score)
    4. Variance of the combined scores (var_score)

    Text: 
    ''' + document + '''

    Summaries: 
    ''' + str(samples) + '''
    

    Please return the computed values for max_score, min_score, mean_score, and var_score.

    [/INST]
    
    '''
    
    encoded_input = tokenizer(ques, return_tensors="pt", truncation=True)
    input_ids = encoded_input.input_ids.to("cuda")
    attention_mask = encoded_input.attention_mask.to("cuda")

    with torch.no_grad():
        output_id = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1000, do_sample=True, temperature=0.1)
    score_output = tokenizer.decode(output_id[0], skip_special_tokens=True)
  
    results[document] = {
        'samples': info['samples'],
        'se': info['se'],
        'be_ent': info['be_ent'],
        'luq': info['luq'],
        'score_output': score_output
    }

with open('eval_3B.json', 'w') as f:
    json.dump(results, f, indent=2)