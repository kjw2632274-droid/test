import os
import time
import random
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from new_multitask.model import MultiHeadKoBART
from new_multitask.dataset_loader import MultiTaskDataset, TaskBatchSampler, build_dataloader
import argparse


def freeze_encoder(model, freeze=True):
    for param in model.encoder.parameters():
        param.requires_grad = not freeze
    print(f"Encoder freeze: {freeze}")

def freeze_decoders(model, train_tasks=None):
    # train_tasks: list of task_names to train, or None for all trainable
    for name, dec in model.decoders.items():
        req_grad = (train_tasks is None) or (name in train_tasks)
        for p in dec.parameters():
            p.requires_grad = req_grad
        for p in model.lm_heads[name].parameters():
            p.requires_grad = req_grad
        print(f"Decoder {name} requires_grad: {req_grad}")

def save_checkpoint(model, optimizer, scheduler, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
    }, os.path.join(out_dir, f"checkpoint_{step}.pt"))
    print(f"[Checkpoint] Saved at step {step}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/processed')
    parser.add_argument('--tokenizer_path', type=str, default='./output/kobart_qa')
    parser.add_argument('--pretrained_model', type=str, default='gogamza/kobart-base-v2')
    parser.add_argument('--output_dir', type=str, default='./output/multitask_ckpt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--train_decoders', type=str, default='all', help='all or comma-separated task_names')
    parser.add_argument('--save_interval_min', type=int, default=30, help='minutes between checkpoints')
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to use for training')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint .pt to resume from')
    parser.add_argument('--include_tasks', type=str, default=None, help='Comma-separated task names to include (default: all)')
    parser.add_argument('--task_max_samples', type=str, default=None, help='Task-specific max samples: task1:num1,task2:num2')
    parser.add_argument('--task_skip_samples', type=str, default=None, help='Task-specific skip samples: task1:num1,task2:num2')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Tokenizer & Data
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    
    # Parse include_tasks
    include_tasks = None
    if args.include_tasks:
        include_tasks = [t.strip() for t in args.include_tasks.split(',')]
    
    # Parse task_max_samples
    task_max_samples = {}
    if args.task_max_samples:
        for pair in args.task_max_samples.split(','):
            if ':' in pair:
                task, num = pair.strip().split(':')
                task_max_samples[task.strip()] = int(num.strip())
    
    # Parse task_skip_samples
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
    
    # Limit dataset size if specified (applies globally after task-specific limits)
    if args.max_samples and args.max_samples < len(dataset):
        print(f"Limiting dataset to {args.max_samples} samples (from {len(dataset)})")
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.max_samples)
        dataset.examples = [dataset.examples[i] for i in indices]
    sampler = TaskBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    dataloader = build_dataloader(dataset, sampler=sampler, num_workers=0)

    # Model
    model = MultiHeadKoBART.from_pretrained(args.pretrained_model)
    model.to(args.device)

    # Freeze encoder if needed
    if args.freeze_encoder:
        freeze_encoder(model, True)
    else:
        freeze_encoder(model, False)

    # Freeze decoders if needed
    if args.train_decoders == 'all':
        freeze_decoders(model, train_tasks=None)
        train_tasks = None
    else:
        train_tasks = [t.strip() for t in args.train_decoders.split(',')]
        freeze_decoders(model, train_tasks=train_tasks)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = args.epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        # Skip optimizer/scheduler loading if training different decoders
        # (parameter groups won't match)
        try:
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
        except (ValueError, KeyError) as e:
            print(f"Warning: Could not load optimizer/scheduler state ({e}). Starting fresh optimizer.")
        if 'step' in ckpt:
            start_step = ckpt['step']
        print(f"Resumed from step {start_step}")
        model.to(args.device)

    # Training loop
    step = start_step
    last_save = time.time()
    total_batches = len(dataloader)
    total_steps = args.epochs * total_batches
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            step += 1
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            task = batch['task']
            # If only one decoder is being trained, force task_name
            task_name = task if train_tasks is None else train_tasks[0]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task_name=task_name)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                percent = 100.0 * (step / total_steps)
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{total_batches} | Step {step}/{total_steps} | {percent:.1f}% | Loss {loss.item():.4f}")
            # Save every N minutes
            if time.time() - last_save > args.save_interval_min * 60:
                save_checkpoint(model, optimizer, scheduler, step, args.output_dir)
                last_save = time.time()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    # Final save
    save_checkpoint(model, optimizer, scheduler, step, args.output_dir)
    print("Training complete.")

if __name__ == '__main__':
    main()
