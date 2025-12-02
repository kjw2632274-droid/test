import os
import time
import random
import json
import torch
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
    parser.add_argument('--task_repeat_factors', type=str, default='style_transfer:3,role_generation:3', help='Task repeat factors for oversampling (e.g., task1:2.5,task2:1). Use "none" to disable.')
    parser.add_argument('--validation_split', type=float, default=1.0/11.0, help='Fraction of data reserved for validation (train:validation = 10:1 by default)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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

    # Parse task_repeat_factors (supports "none" to disable)
    task_repeat_factors = {}
    if args.task_repeat_factors and args.task_repeat_factors.lower() != 'none':
        for pair in args.task_repeat_factors.split(','):
            if ':' in pair:
                task, num = pair.strip().split(':')
                task_repeat_factors[task.strip()] = float(num.strip())
    
    train_dataset = MultiTaskDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_input_length=256,
        max_target_length=128,
        include_tasks=include_tasks,
        drop_empty_target=True,
        task_max_samples=task_max_samples,
        task_skip_samples=task_skip_samples,
        task_repeat_factors=task_repeat_factors,
        split='train',
        validation_split=args.validation_split,
        seed=args.seed,
    )
    
    # Limit dataset size if specified (applies globally after task-specific limits)
    if args.max_samples and args.max_samples < len(train_dataset):
        print(f"Limiting dataset to {args.max_samples} samples (from {len(train_dataset)})")
        random.seed(args.seed)
        indices = random.sample(range(len(train_dataset)), args.max_samples)
        train_dataset.examples = [train_dataset.examples[i] for i in indices]
    train_sampler = TaskBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_dataloader = build_dataloader(train_dataset, sampler=train_sampler, num_workers=0)
    if len(train_dataloader) == 0:
        raise ValueError('Training dataloader is empty. Check task filters, split ratio, and batch size.')

    val_dataset = None
    val_dataloader = None
    if 0.0 < args.validation_split < 1.0:
        val_dataset = MultiTaskDataset(
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            max_input_length=256,
            max_target_length=128,
            include_tasks=include_tasks,
            drop_empty_target=True,
            task_max_samples=task_max_samples,
            task_skip_samples=task_skip_samples,
            task_repeat_factors=task_repeat_factors,
            split='validation',
            validation_split=args.validation_split,
            seed=args.seed,
        )
        if len(val_dataset) > 0:
            val_sampler = TaskBatchSampler(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            val_dataloader = build_dataloader(val_dataset, sampler=val_sampler, num_workers=0, drop_last=False)

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
    total_steps = args.epochs * len(train_dataloader)
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
    total_batches = len(train_dataloader)
    total_steps = args.epochs * total_batches
    train_history = []
    val_history = []
    run_id = time.strftime('%Y%m%d-%H%M%S')

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            step += 1
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            task = batch['task']
            # Always use the batch's actual task (dataset already filtered by include_tasks)
            task_name = task
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task_name=task_name)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                percent = 100.0 * (step / total_steps)
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_idx+1}/{total_batches} | Step {step}/{total_steps} | {percent:.1f}% | Loss {loss.item():.4f}")
                train_history.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'step': step,
                    'loss': float(loss.item()),
                    'timestamp': time.time(),
                })
            # Save every N minutes
            if time.time() - last_save > args.save_interval_min * 60:
                save_checkpoint(model, optimizer, scheduler, step, args.output_dir)
                last_save = time.time()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
        if val_dataloader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_dataloader:
                    input_ids = val_batch['input_ids'].to(args.device)
                    attention_mask = val_batch['attention_mask'].to(args.device)
                    labels = val_batch['labels'].to(args.device)
                    task_name = val_batch['task']
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, task_name=task_name)
                    val_losses.append(outputs['loss'].item())
            if val_losses:
                avg_val = sum(val_losses) / len(val_losses)
                print(f"Epoch {epoch+1}/{args.epochs} | Validation Loss {avg_val:.4f}")
                val_history.append({
                    'epoch': epoch + 1,
                    'step': step,
                    'loss': float(avg_val),
                    'timestamp': time.time(),
                })
            model.train()
    # Final save
    save_checkpoint(model, optimizer, scheduler, step, args.output_dir)

    log_payload = {
        'config': vars(args),
        'train_history': train_history,
        'val_history': val_history,
        'total_steps': total_steps,
        'start_step': start_step,
        'final_step': step,
        'log_created_at': run_id,
    }
    log_path = os.path.join(args.output_dir, f'training_log_{run_id}.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_payload, f, ensure_ascii=False, indent=2)
    print(f"[TrainingLog] Saved to {log_path}")
    print("Training complete.")

if __name__ == '__main__':
    main()
