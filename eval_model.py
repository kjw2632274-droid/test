"""
Evaluate model.py (MultiHeadKoBART)

Usage:
    python -m new_multitask.eval_model --ckpt_path output/multitask_ckpt/checkpoint_143247.pt --num_samples 20
"""

import argparse
import json
import os
import random

import torch
from transformers import PreTrainedTokenizerFast

# Import from updated module path
from new_multitask.model import MultiHeadKoBART
from new_multitask.dataset_loader import MultiTaskDataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate model.py')
    
    # Model
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, default='gogamza/kobart-base-v2')
    parser.add_argument('--tokenizer_path', type=str, default='gogamza/kobart-base-v2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--task', type=str, default='all', help='Task to evaluate or "all"')
    parser.add_argument('--num_samples', type=int, default=20, help='Total samples to evaluate')
    parser.add_argument('--print_per_task', type=int, default=None, help='Number of samples to print per task (optional)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Generation
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--repetition_penalty', type=float, default=1.5, help='>1.0이면 반복 감소')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=2, help='해당 n-gram 반복 차단')
    
    # Output
    parser.add_argument('--save_preds', type=str, default='output/eval_model.jsonl')
    
    args = parser.parse_args()
    
    print('='*70)
    print('EVALUATING MODEL.PY')
    print('='*70)
    
    # Load tokenizer
    print(f'\nLoading tokenizer: {args.tokenizer_path}')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # Load dataset
    print(f'Loading dataset: {args.data_dir}')
    dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=64
    )
    
    # Load model
    print(f'Loading model: {args.pretrained_model}')
    model = MultiHeadKoBART.from_pretrained(args.pretrained_model)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.ckpt_path}')
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    model.to(args.device)
    model.eval()
    
    # Print model info
    model.print_model_info()
    
    # Select samples
    random.seed(args.seed)
    
    if args.task == 'all':
        # Sample evenly from all tasks
        tasks = ['qa', 'role_generation', 'style_transfer', 'dialogue_summarization']
        samples_per_task = args.num_samples // len(tasks)
        
        selected = []
        for task in tasks:
            task_indices = [i for i, ex in enumerate(dataset.examples) if ex['task'] == task]
            if len(task_indices) > 0 and samples_per_task > 0:
                sampled = random.sample(task_indices, min(samples_per_task, len(task_indices)))
                selected.extend(sampled)
        
        # Fill remaining slots randomly
        remaining = args.num_samples - len(selected)
        if remaining > 0:
            all_indices = set(range(len(dataset)))
            available = list(all_indices - set(selected))
            if available:
                selected.extend(random.sample(available, min(remaining, len(available))))
    else:
        # Sample from specific task
        task_indices = [i for i, ex in enumerate(dataset.examples) if ex['task'] == args.task]
        selected = random.sample(task_indices, min(args.num_samples, len(task_indices))) if task_indices else []
    
    device = torch.device(args.device)
    
    # Evaluate
    print(f'\nEvaluating {len(selected)} samples...\n')
    
    results = []
    total_loss = 0.0
    task_losses = {}
    
    # Track printing limits per task
    print_limits = {t: (args.print_per_task if args.print_per_task is not None else None) for t in ['qa','role_generation','style_transfer','dialogue_summarization']}

    with torch.no_grad():
        for idx_num, idx in enumerate(selected, 1):
            ex = dataset.examples[idx]
            task = ex['task']
            item = dataset[idx]
            
            # Get input/target text
            inp_text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
            tgt_text = tokenizer.decode(item['labels'], skip_special_tokens=True)
            
            # Strip task label
            stripped = inp_text
            for prefix in [f'[{task}]', f'[{task}] ']:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):].lstrip()
                    break
            
            # Encode
            enc = tokenizer(stripped, return_tensors='pt', truncation=True, max_length=512, add_special_tokens=False)
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Compute loss
            labels = item['labels'].unsqueeze(0).to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task_name=task,
            )
            loss = float(output['loss'].item())
            total_loss += loss
            task_losses.setdefault(task, []).append(loss)
            
            # Generate
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_name=task,
                max_length=args.max_length,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            
            # Conditional printing per task
            can_print = True
            if args.task == 'all' and print_limits.get(task) is not None:
                # Count prior prints for this task
                prior_prints = task_losses.get(task, [])
                can_print = (len(prior_prints) <= print_limits[task] - 1)
            if args.task != 'all' and args.print_per_task is not None:
                prior_prints = task_losses.get(task, [])
                can_print = (len(prior_prints) <= args.print_per_task - 1)
            if can_print:
                print(f'[{idx_num}/{len(selected)}] Task: {task} | Loss: {loss:.4f}')
                print(f'Input:  {stripped}')
                print(f'Target: {tgt_text}')
                print(f'Pred:   {pred_text}')
                print()
            
            # Save result
            results.append({
                'task': task,
                'input': stripped,
                'target': tgt_text,
                'prediction': pred_text,
                'loss': loss,
            })
    
    # Summary
    print('='*70)
    print('EVALUATION RESULTS')
    print('='*70)
    print(f'Total samples: {len(selected)}')
    print(f'Average loss:  {total_loss / len(selected) if selected else 0.0:.4f}')
    print()
    print('Per-task results:')
    for task in sorted(task_losses.keys()):
        losses = task_losses[task]
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        print(f'  {task:24s}: {len(losses):3d} samples, avg loss: {avg_loss:.4f}')
    
    # Save predictions
    os.makedirs(os.path.dirname(args.save_preds), exist_ok=True)
    with open(args.save_preds, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\nPredictions saved to: {args.save_preds}')
    print('='*70)


if __name__ == '__main__':
    main()
