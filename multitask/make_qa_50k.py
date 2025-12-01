"""
Extract 50k QA samples from qa_generation.jsonl and convert to qa_train.jsonl format
"""
import json
import random
from pathlib import Path

def load_qa_generation(file_path):
    """Load qa_generation.jsonl and extract question/answer pairs"""
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Extract question from input field
                input_text = item.get('input', '')
                # Question is after [Q] marker
                if '[Q]' in input_text:
                    question = input_text.split('[Q]')[-1].strip()
                    answer = item.get('target', '').strip()
                    
                    if question and answer:
                        qa_pairs.append({
                            'question': question,
                            'answer': answer
                        })
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    return qa_pairs

def main():
    input_file = Path('./data/processed/qa_generation.jsonl')
    output_file = Path('./data/processed/qa_train.jsonl')
    
    print("Loading qa_generation.jsonl...")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    all_qa = load_qa_generation(input_file)
    print(f"Loaded {len(all_qa)} QA pairs from qa_generation.jsonl")
    
    # Sample or duplicate to reach 50k
    target_count = 50000
    if len(all_qa) >= target_count:
        # Random sample
        random.seed(42)
        selected_qa = random.sample(all_qa, target_count)
        print(f"Randomly sampled {target_count} QA pairs")
    else:
        # Duplicate with shuffle to reach target
        random.seed(42)
        selected_qa = []
        while len(selected_qa) < target_count:
            remaining = target_count - len(selected_qa)
            to_add = min(remaining, len(all_qa))
            shuffled = random.sample(all_qa, to_add)
            selected_qa.extend(shuffled)
        print(f"Duplicated and shuffled to reach {target_count} QA pairs")
    
    # Write to JSONL in qa_train format
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in selected_qa:
            json.dump(qa, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(selected_qa)} QA pairs to {output_file}")
    print("Done!")

if __name__ == '__main__':
    random.seed(42)
    main()
