import json
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def compute_semantic_similarities(samples, tokenizer, model):
    """
    Compute semantic similarities between samples using MNLI model.
    
    Args:
        samples (list): List of text samples
        tokenizer: Hugging Face tokenizer
        model: MNLI classification model
    
    Returns:
        list: Semantic set IDs for samples
    """
    semantic_set_ids = list(range(len(samples)))
    device = model.device

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            # Use separator token to concatenate samples
            input_text = samples[i] + ' [SEP] ' + samples[j]
            
            # Tokenize with error handling and proper padding
            try:
                encoded_input = tokenizer(
                    input_text, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                ).to(device)
                
                with torch.no_grad():
                    output = model(**encoded_input)
                    predicted_label = torch.argmax(output.logits, dim=1).item()
                
                # If not contradiction, merge semantic sets
                if predicted_label != 0:
                    semantic_set_ids[j] = semantic_set_ids[i]
            
            except Exception as e:
                print(f"Error processing samples {i} and {j}: {e}")
                # Keep original semantic set if processing fails
                continue

    return semantic_set_ids

def compute_likelihoods(samples, tokenizer, model, batch_size=8):
    """
    Compute negative log-likelihoods for samples.
    
    Args:
        samples (list): List of text samples
        tokenizer: Hugging Face tokenizer
        model: Language model for likelihood computation
        batch_size (int): Number of samples to process in each batch
    
    Returns:
        torch.Tensor: Negative log-likelihoods for samples
    """
    device = model.device
    neg_log_likelihoods = []

    for i in range(0, len(samples), batch_size):
        try:
            batch = samples[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                batch_neg_log_likelihoods = -log_probs.sum(dim=-1).cpu()
                
                neg_log_likelihoods.extend(batch_neg_log_likelihoods.tolist())
        
        except Exception as e:
            print(f"Error computing likelihoods for batch starting at index {i}: {e}")
            # Skip problematic batches
            continue

    return torch.tensor(neg_log_likelihoods)

def get_predictive_entropy_over_concepts(log_likelihoods, semantic_set_ids):
    """
    Compute predictive entropy over semantic concepts.
    
    Args:
        log_likelihoods (torch.Tensor): Log-likelihoods for samples
        semantic_set_ids (list): Semantic set identifiers for samples
    
    Returns:
        float: Predictive entropy
    """
    unique_concepts = set(semantic_set_ids)
    aggregated_likelihoods = []

    for concept in unique_concepts:
        concept_indices = [i for i, sid in enumerate(semantic_set_ids) if sid == concept]
        
        if concept_indices:  # Check if the list is not empty
            concept_likelihood = torch.logsumexp(log_likelihoods[concept_indices], dim=0)
            aggregated_likelihoods.append(concept_likelihood)

    if aggregated_likelihoods:
        aggregated_likelihoods = torch.stack(aggregated_likelihoods)
        entropy = -torch.sum(torch.exp(aggregated_likelihoods) * aggregated_likelihoods)
        return entropy.item()
    else:
        return 0.0  # Return zero if no valid concepts

def process_json_file(file_path, output_path=None):
    """
    Process JSON file with semantic and entropy analysis.
    
    Args:
        file_path (str): Input JSON file path
        output_path (str, optional): Output JSON file path
    
    Returns:
        dict: Processed results
    """
    # Ensure GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models with error handling
    try:
        similarity_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        similarity_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)
        
        likelihood_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        likelihood_model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-350m").to(device)
    except Exception as e:
        print(f"Model loading error: {e}")
        return {}

    # Load JSON data
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return {}

    results = {}
    for document, info in data.items():
        samples = info.get('samples', [])
        
        if not samples:
            print(f"No samples found for document: {document}")
            continue

        try:
            # Compute semantic similarities and likelihoods
            semantic_set_ids = compute_semantic_similarities(samples, similarity_tokenizer, similarity_model)
            neg_log_likelihoods = compute_likelihoods(samples, likelihood_tokenizer, likelihood_model)
            
            predictive_entropy_over_concepts = get_predictive_entropy_over_concepts(
                -neg_log_likelihoods, 
                semantic_set_ids
            )

            results[document] = {
                'samples': samples,
                'se': info.get('se', None),
                'be_ent': float(predictive_entropy_over_concepts)
            }
        
        except Exception as e:
            print(f"Error processing document {document}: {e}")
            continue

    # Optional: Save results to file
    if output_path:
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results written to {output_path}")
        except IOError as e:
            print(f"File writing error: {e}")

    return results

def main():
    file_path = 'cp_generations_samsum_1.json'
    output_path = 'output3.json'
    processed_data = process_json_file(file_path, output_path)

if __name__ == '__main__':
    main()