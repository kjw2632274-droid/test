import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from new_multitask.model import MultiHeadKoBART
from new_multitask.dataset_loader import MultiTaskDataset, TaskBatchSampler, build_dataloader
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--tokenizer_path', type=str, default='./output/kobart_qa')
    parser.add_argument('--pretrained_model', type=str, default='gogamza/kobart-base-v2')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Optional: path to model checkpoint .pt')
    parser.add_argument('--task', type=str, default='all', help='all or one of: qa, style_transfer, dialogue_summarization, role_generation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--task_max_samples', type=str, default=None, help='Task-specific max samples: task1:num1,task2:num2')
    parser.add_argument('--task_skip_samples', type=str, default=None, help='Task-specific skip samples: task1:num1,task2:num2')
    # Generation params
    parser.add_argument('--gen_max_length', type=int, default=32)
    parser.add_argument('--gen_num_beams', type=int, default=1)
    parser.add_argument('--gen_repetition_penalty', type=float, default=2.5)
    parser.add_argument('--gen_no_repeat_ngram_size', type=int, default=2)
    parser.add_argument('--gen_length_penalty', type=float, default=1.0)
    parser.add_argument('--gen_early_stopping', action='store_true', help='Enable early stopping for beam search')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    # Only load the relevant task for evaluation
    include_tasks = [args.task] if args.task != 'all' else None
    # Parse task_max_samples and task_skip_samples for eval (optional)
    task_max_samples = {}
    if args.task_max_samples:
        for pair in args.task_max_samples.split(','):
            if ':' in pair:
                task, num = pair.strip().split(':')
                task_max_samples[task.strip()] = int(num.strip())

    task_skip_samples = {}
    if args.task_skip_samples:
        for pair in args.task_skip_samples.split(','):
            if ':' in pair:
                task, num = pair.strip().split(':')
                task_skip_samples[task.strip()] = int(num.strip())

    dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_input_length=256,
        max_target_length=128,
        include_tasks=include_tasks,
        drop_empty_target=True,
        task_max_samples=task_max_samples,
        task_skip_samples=task_skip_samples,
    )
    sampler = TaskBatchSampler(dataset, batch_size=args.batch_size, shuffle=False)
    dataloader = build_dataloader(dataset, sampler=sampler, num_workers=2)

    model = MultiHeadKoBART.from_pretrained(args.pretrained_model)
    if args.ckpt_path:
        state = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state['model'], strict=False)
        print(f"Loaded checkpoint: {args.ckpt_path}")
    model.to(args.device)
    model.eval()

    total_loss = 0.0
    total_count = 0
    task_losses = defaultdict(float)
    task_counts = defaultdict(int)
    shown = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            task = batch['task']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task_name=task)
            loss = outputs['loss']
            bsz = input_ids.size(0)
            total_loss += loss.item() * bsz
            total_count += bsz
            task_losses[task] += loss.item() * bsz
            task_counts[task] += bsz
            # Print a few predictions
            if shown < 5:
                preds = model.generate(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    task_name=task, 
                    max_length=args.gen_max_length,
                    num_beams=args.gen_num_beams,
                    repetition_penalty=args.gen_repetition_penalty,
                    no_repeat_ngram_size=args.gen_no_repeat_ngram_size,
                    length_penalty=args.gen_length_penalty,
                    early_stopping=args.gen_early_stopping,
                )
                for i in range(min(bsz, 2)):
                    inp = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    tgt = tokenizer.decode(labels[i], skip_special_tokens=True)
                    pred = tokenizer.decode(preds[i], skip_special_tokens=True)
                    print(f"\n[{task}]\nInput: {inp}\nTarget: {tgt}\nPred:   {pred}")
                    shown += 1
            if args.max_samples and total_count >= args.max_samples:
                break
    avg_loss = total_loss / max(1, total_count)
    print(f"\n[Eval] Total samples: {total_count}  Avg loss: {avg_loss:.4f}")
    for t in task_losses:
        print(f"  {t}: {task_counts[t]} samples, avg loss: {task_losses[t]/max(1,task_counts[t]):.4f}")

if __name__ == '__main__':
    main()
