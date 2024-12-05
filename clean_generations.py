# import json
# import re

# def clean_sample_text(sample_text):
#     # Find the positions of [INST] and [/INST]
#     inst_start = sample_text.find('[INST]')
#     inst_end = sample_text.find('[/INST]')
    
#     # Case 1 & 2: Extract text between [/INST] and [INST] or [INST] and [/INST]
#     if inst_start != -1 and inst_end != -1:
#         start = min(inst_start, inst_end)
#         end = max(inst_start, inst_end)
#         cleaned_text = sample_text[start+6:end].strip()
    
#     # Case 3: Extract text before [INST] if it's after the text
#     elif inst_start != -1:
#         cleaned_text = sample_text[:inst_start].strip()
    
#     # Case 4: Extract text before [/INST] if it's after the text
#     elif inst_end != -1:
#         cleaned_text = sample_text[:inst_end].strip()
    
#     # Case 5 & 6: Extract text before the first full stop if [INST] or [/INST] is before the text
#     elif sample_text.startswith('[INST]') or sample_text.startswith('[/INST]'):
#         match = re.search(r'\.(.*?)\[(?:INST|/INST)\]', sample_text)
#         if match:
#             cleaned_text = match.group(1).strip()
#         else:
#             cleaned_text = sample_text
    
#     # Case 7: Leave text as is if no markers are found
#     else:
#         cleaned_text = sample_text
    
#     return cleaned_text

# def process_json(input_json):
#     with open(input_json, 'r') as f:
#         data = json.load(f)

#     for key, document in data.items():
#         document['samples'] = [clean_sample_text(sample) for sample in document['samples']]
    
#     return data

# def save_cleaned_json(data, output_json):
#     with open(output_json, 'w') as f:
#         json.dump(data, f, indent=4)

# if __name__ == '__main__':
#     input_json = 'sftp/cp_generations_samsum_.json'
#     output_json = 'sftp/cp_generations_samsum.json'

#     cleaned_data = process_json(input_json)
#     save_cleaned_json(cleaned_data, output_json)

import json
import re

def clean_sample_text(sample_text):
    inst_start = sample_text.find('[INST]')
    inst_end = sample_text.find('[/INST]')
    
    if inst_start != -1 and inst_end != -1:
        start = min(inst_start, inst_end)
        end = max(inst_start, inst_end)
        cleaned_text = sample_text[start+6:end].strip()
    elif inst_start != -1:
        cleaned_text = sample_text[:inst_start].strip()
    elif inst_end != -1:
        cleaned_text = sample_text[:inst_end].strip()
    elif sample_text.startswith('[INST]') or sample_text.startswith('[/INST]'):
        match = re.search(r'\.(.*?)\[(?:INST|/INST)\]', sample_text)
        if match:
            cleaned_text = match.group(1).strip()
        else:
            cleaned_text = sample_text
    else:
        cleaned_text = sample_text
    
    return cleaned_text

def process_json(input_json):
    with open(input_json, 'r') as f:
        data = json.load(f)

    for key, document in data.items():
        # Clean each sample and remove empty strings
        document['samples'] = [clean_sample_text(sample) for sample in document['samples'] if clean_sample_text(sample)]
    
    return data

def save_cleaned_json(data, output_json):
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    input_json = 'sftp/cp_generations_samsum_.json'
    output_json = 'sftp/cp_generations_samsum.json'

    cleaned_data = process_json(input_json)
    save_cleaned_json(cleaned_data, output_json)