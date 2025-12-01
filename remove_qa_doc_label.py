"""
Remove [doc] label from qa_generation.jsonl
"""
import json
from pathlib import Path

def remove_doc_label(input_file, output_file):
    """
    Remove [doc] label from input field in JSONL file
    
    Example:
    Input: {"input": "[qa] [doc] question", "target": "answer"}
    Output: {"input": "[qa] question", "target": "answer"}
    """
    processed_count = 0
    removed_count = 0
    
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                input_text = data.get('input', '')
                
                # Remove [doc] label
                if '[doc]' in input_text.lower():
                    # Remove case-insensitive [doc] or [DOC]
                    import re
                    input_text = re.sub(r'\[doc\]\s*', '', input_text, flags=re.IGNORECASE)
                    data['input'] = input_text
                    removed_count += 1
                
                # Write processed line
                json.dump(data, fout, ensure_ascii=False)
                fout.write('\n')
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count} lines...")
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total lines processed: {processed_count}")
    print(f"Lines with [doc] removed: {removed_count}")

def main():
    input_file = Path('./data/processed/qa_generation.jsonl')
    output_file = Path('./data/processed/qa_generation_no_doc.jsonl')
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        return
    
    remove_doc_label(input_file, output_file)
    print(f"\nOutput saved to: {output_file}")

if __name__ == '__main__':
    main()
