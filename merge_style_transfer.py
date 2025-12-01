"""
Merge style_transfer datasets: smilestyle_dataset.tsv + train_manual.jsonl
"""
import json
from pathlib import Path

def load_smilestyle_tsv(file_path):
    """Load formal->informal data from smilestyle_dataset.tsv"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                formal = parts[0].strip()
                informal = parts[1].strip()
                if formal and informal:
                    data.append({
                        'formal': f'[style_transfer] {formal}',
                        'casual': informal
                    })
    return data

def load_train_manual(file_path):
    """Load train_manual.jsonl (already has [style_transfer] prefix)"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'formal' in item and 'casual' in item:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    return data

def main():
    # Load both datasets
    smilestyle_file = Path('./data/style_transfer/korean_smile_style_dataset/smilestyle_dataset.tsv')
    train_manual_file = Path('./data/processed/train_manual.jsonl')
    output_file = Path('./data/processed/style_transfer_formal_informal_tagged.jsonl')
    
    print("Loading smilestyle_dataset.tsv...")
    smilestyle_data = load_smilestyle_tsv(smilestyle_file)
    print(f"Loaded {len(smilestyle_data)} samples from smilestyle")
    
    print("Loading train_manual.jsonl...")
    train_manual_data = load_train_manual(train_manual_file)
    print(f"Loaded {len(train_manual_data)} samples from train_manual")
    
    # Merge
    all_data = smilestyle_data + train_manual_data
    print(f"Total samples: {len(all_data)}")
    
    # Write to output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8-sig') as f:
        for item in all_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Saved to {output_file}")
    print("Done!")

if __name__ == '__main__':
    main()
