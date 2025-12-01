import json

input_file = "data/processed/style_transfer_formal_informal_tagged.jsonl"
output_file = "data/processed/style_transfer_bidirectional.jsonl"

original_samples = []
reversed_samples = []

# Read original data
with open(input_file, 'r', encoding='utf-8-sig') as f:
    for line in f:
        data = json.loads(line.strip())
        # Convert to input/target format
        original = {
            "input": data["formal"],
            "target": data["casual"]
        }
        original_samples.append(original)
        
        # Create reversed sample (casual -> formal)
        reversed_data = {
            "input": "[style_transfer] " + data["casual"],
            "target": data["formal"].replace("[style_transfer] ", "")
        }
        reversed_samples.append(reversed_data)

# Write all samples (original + reversed)
with open(output_file, 'w', encoding='utf-8') as f:
    for data in original_samples:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    for data in reversed_samples:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Original samples: {len(original_samples)}")
print(f"Reversed samples: {len(reversed_samples)}")
print(f"Total samples: {len(original_samples) + len(reversed_samples)}")
print(f"Output file: {output_file}")
