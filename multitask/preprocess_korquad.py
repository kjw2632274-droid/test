"""
Preprocess KorQuAD dataset for QA task training
Converts korquad1_train.json and korquad1_validation.json to JSONL format
"""
import json
from pathlib import Path

def preprocess_korquad(input_file, output_file):
    """
    Convert KorQuAD format to training format
    
    Input format: {"context": "...", "question": "...", "answers": {"text": ["..."], ...}}
    Output format: {"input": "{question}", "target": "..."}
    """
    samples = []
    
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                context = data.get('context', '').strip()
                question = data.get('question', '').strip()
                answers = data.get('answers', {})
                
                # Extract first answer text
                answer_texts = answers.get('text', [])
                if not answer_texts:
                    continue
                
                answer = answer_texts[0].strip()
                
                # Skip if any field is empty
                if not question or not answer:
                    continue
                
                # Construct input: {question} ([qa] 라벨 제거)
                input_text = question
                
                samples.append({
                    'input': input_text,
                    'target': answer
                })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    # Write to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved {len(samples)} samples to {output_file}")
    return len(samples)

def main():
    data_dir = Path('./data/qa/korquad')
    output_dir = Path('./data/processed')
    
    # Process training set
    train_input = data_dir / 'korquad1_train.json'
    train_output = output_dir / 'korquad_train.jsonl'
    
    if train_input.exists():
        train_count = preprocess_korquad(train_input, train_output)
    else:
        print(f"Warning: {train_input} not found")
        train_count = 0
    
    # Process validation set
    valid_input = data_dir / 'korquad1_validation.json'
    valid_output = output_dir / 'korquad_valid.jsonl'
    
    if valid_input.exists():
        valid_count = preprocess_korquad(valid_input, valid_output)
    else:
        print(f"Warning: {valid_input} not found")
        valid_count = 0
    
    print("\n=== Preprocessing Complete ===")
    print(f"Train samples: {train_count}")
    print(f"Valid samples: {valid_count}")
    print(f"Total samples: {train_count + valid_count}")
    print("\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {valid_output}")

if __name__ == '__main__':
    main()
