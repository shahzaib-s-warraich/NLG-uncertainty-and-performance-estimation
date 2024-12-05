import json
import numpy as np
from itertools import combinations
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

class LUQ_vllm:
    def __init__(self, model="llama3-8b-instruct", method="binary", abridged=False):
        if model == "llama3-8b-instruct":
            model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif model == "Llama-3.2-3B-Instruct":
            model_path = "meta-llama/Llama-3.2-3B-Instruct" 
        else:
            raise ValueError("Model not supported")

        self.method = method
        self.abridged = abridged
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.sampling_params = SamplingParams(
            n=1, temperature=0, top_p=0.9, max_tokens=5,
            stop_token_ids=[self.tokenizer.eos_token_id], skip_special_tokens=True,
        )

        self.llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.5, download_dir='/home/shahzaib/exps/se_exps')
        
        self.prompt_template = "Context: {context}\n\nSentence: {sentence}\n\nIs the sentence supported by the context above? Answer Yes or No.\n\nAnswer: "
        self.text_mapping = {'yes': 1, 'no': 0, 'n/a': 0.5}

        self.not_defined_text = set()

    def completion(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return outputs

    def text_postprocessing(self, text):
        text = text.lower().strip()
        if text[:3] == 'yes':
            text = 'yes'
        elif text[:2] == 'no':
            text = 'no'
        else:
            text = 'n/a'
        
        if text == 'n/a' and text not in self.not_defined_text:
            print(f"warning: {text} not defined")
            self.not_defined_text.add(text)
        
        return self.text_mapping[text]

    def predict_pair(self, sample1, sample2):
        prompt_text = self.prompt_template.format(context=sample1, sentence=sample2)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        output = self.completion([prompt])[0]
        generate_text = output.outputs[0].text
        score = self.text_postprocessing(generate_text)
        
        return score

def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    luq_calculator = LUQ_vllm(model="Llama-3.2-3B-Instruct", method="binary", abridged=False)
    
    results = {}
    for document, info in tqdm(data.items()):
        samples = info['samples']
        
        luq_scores = []
        for sample1, sample2 in combinations(samples, 2):
            score1 = luq_calculator.predict_pair(sample1, sample2)
            score2 = luq_calculator.predict_pair(sample2, sample1)
            avg_score = (score1 + score2) / 2
            luq_scores.append(avg_score)
        
        results[document] = {
            'samples': info['samples'],
            'se': info['se'],
            'be_ent': info['be_ent'],
            'luq': np.mean(luq_scores)
        }
        # results[document] = {
        #     'samples': info['samples'],
        #     'luq': np.mean(luq_scores)
        # }

    return results

def export_to_json(processed_results, output_file):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(output_file, 'w') as jsonfile:
        json.dump(processed_results, jsonfile, cls=NumpyEncoder, indent=2)

if __name__ == '__main__':
    json_file_path = 'cp_generations_samsum_2.json'  # Replace with your JSON file path
    processed_results = process_json_file(json_file_path)

    output_file = 'cp_generations_samsum_3.json'
    export_to_json(processed_results, output_file)
    print(f'Processed results exported to {output_file}')