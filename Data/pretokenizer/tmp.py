import json

parsed_dataset_path = '../../../parsed_dataset.json'

# Load the entire JSON array
with open(parsed_dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Print first 10 entries
for i, example in enumerate(data[:10]):
    print(f"Example {i + 1}:\n{json.dumps(example["parsed"], indent=2)}\n{'-'*40}\n")
