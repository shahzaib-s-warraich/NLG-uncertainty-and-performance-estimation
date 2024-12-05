import json
import numpy as np
from nltk.tokenize import word_tokenize
import evaluate

bertscore = evaluate.load("bertscore")

def calc_bertscore(gen_a: str, gen_b: str) -> float:
    return bertscore.compute(predictions=[gen_a], references=[gen_b], lang='en')['f1'][0]

def calc_bertscore_length(gen_a: str, gen_b: str) -> float:
    ref, cand = (gen_a, gen_b) if len(gen_a) > len(gen_b) else (gen_b, gen_a)
    try:
        length_pen = np.exp(1 - len(word_tokenize(ref)) / len(word_tokenize(cand)))
        return length_pen * calc_bertscore(gen_a, gen_b)
    except ZeroDivisionError:
        print(f"{gen_a} \n {gen_b}")
        return 0.0

def calculate_self_alignment(samples):
    score_matrix = np.zeros((len(samples), len(samples)))
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            score = calc_bertscore_length(samples[i], samples[j])
            score_matrix[i, j] = score_matrix[j, i] = score

    np.fill_diagonal(score_matrix, np.nan)
    self_alignment = np.nanmean(score_matrix)
    return self_alignment

def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    results = {}
    for document, info in data.items():
        samples = info['samples']
        self_alignment_score = calculate_self_alignment(samples)
        
        results[document] = {
            'samples': info['samples'],
            'se': self_alignment_score
        }

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
    json_file_path = '0.3_generations_xsum.json'  # Replace with your JSON file path
    processed_results = process_json_file(json_file_path)

    output_file = 'res_0.3_generations_xsum.json'
    export_to_json(processed_results, output_file)
    print(f'Processed results exported to {output_file}')

    # Optional: Print results to console
    for document, info in processed_results.items():
        print(f"Document: {document}")
        print("Samples:")
        print(f"Self-alignment score: {info['self_alignment']}")
        for sample in info['samples']:
            print(f"  - {sample}")
        print("\n")