"""Evaluate MultiHeadKoBART on fixed sample counts and plot batch loss curves."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from transformers import PreTrainedTokenizerFast

from new_multitask.dataset_loader import MultiTaskDataset, TaskBatchSampler, build_dataloader
from new_multitask.model import MultiHeadKoBART, TASKS


DEFAULT_TASK_SAMPLES = {
    "dialogue_summarization": 10_000,
    "qa": 6_000,
    "role_generation": 378,
    "style_transfer": 15_000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model and plot loss curves")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--pretrained_model", type=str, default="gogamza/kobart-base-v2", help="Base pretrained model name or path")
    parser.add_argument("--tokenizer_path", type=str, default="gogamza/kobart-base-v2", help="Tokenizer name or path")
    parser.add_argument("--data_dir", type=str, default="processed/processed", help="Directory with task jsonl files")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--output_dir", type=str, default="output/eval_curve", help="Directory to store artifacts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument(
        "--include_tasks",
        type=str,
        default=None,
        help="Comma-separated task names to evaluate (default: all)",
    )
    parser.add_argument("--task_samples", type=str, default=None, help="Override task sample counts, e.g. qa:4000,style_transfer:8000")
    parser.add_argument("--max_input_length", type=int, default=256, help="Tokenizer input max length")
    parser.add_argument("--max_target_length", type=int, default=128, help="Tokenizer target max length")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-batch logging")
    return parser.parse_args()


def parse_task_samples(arg: str | None, include_tasks: List[str]) -> Dict[str, int]:
    samples = DEFAULT_TASK_SAMPLES.copy()
    if include_tasks:
        samples = {k: v for k, v in samples.items() if k in include_tasks}
    if not arg:
        return samples
    for pair in arg.split(","):
        if not pair.strip():
            continue
        if ":" not in pair:
            raise ValueError(f"Invalid task_samples entry '{pair}'. Expected task:num")
        task, value = pair.split(":", 1)
        task = task.strip()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Allowed: {', '.join(TASKS)}")
        samples[task] = int(value.strip())
    return samples


def build_eval_loader(
    data_dir: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    task_samples: Dict[str, int],
    max_input: int,
    max_target: int,
    num_workers: int,
) -> Tuple[MultiTaskDataset, torch.utils.data.DataLoader]:
    dataset = MultiTaskDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_input_length=max_input,
        max_target_length=max_target,
        include_tasks=list(task_samples.keys()),
        task_max_samples=task_samples,
        task_repeat_factors={},
        split="all",
        validation_split=0.0,
        drop_empty_target=True,
    )

    sampler = TaskBatchSampler(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    loader = build_dataloader(dataset, sampler=sampler, num_workers=num_workers, drop_last=False)
    return dataset, loader


def load_model(args: argparse.Namespace) -> tuple[MultiHeadKoBART, PreTrainedTokenizerFast]:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    model = MultiHeadKoBART.from_pretrained(args.pretrained_model)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(args.device)
    model.eval()
    return model, tokenizer


def evaluate(model: MultiHeadKoBART, loader, device: str, quiet: bool) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    step = 0
    with torch.no_grad():
        for batch in loader:
            step += 1
            task = batch["task"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                task_name=task,
            )
            loss = float(outputs["loss"].item())
            records.append({
                "step": step,
                "task": task,
                "loss": loss,
                "batch_size": int(input_ids.size(0)),
            })
            if not quiet and step % 20 == 0:
                print(f"Step {step}: task={task} loss={loss:.4f}")
    return records


def plot_losses(records: List[Dict[str, float]], out_path: str) -> None:
    if not records:
        print("No records to plot.")
        return
    plt.figure(figsize=(12, 6))
    by_task: Dict[str, Tuple[List[int], List[float]]] = {}
    for rec in records:
        task = rec["task"]
        if task not in by_task:
            by_task[task] = ([], [])
        by_task[task][0].append(rec["step"])
        by_task[task][1].append(rec["loss"])

    for task, (steps, losses) in by_task.items():
        plt.plot(steps, losses, marker="o", linewidth=1.0, markersize=3, label=task)

    plt.xlabel("Batch index")
    plt.ylabel("Loss")
    plt.title("Evaluation loss per batch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved loss curve to {out_path}")


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for task in TASKS:
        losses = [r["loss"] for r in records if r["task"] == task]
        if not losses:
            continue
        summary[task] = {
            "batches": len(losses),
            "mean_loss": float(sum(losses) / len(losses)),
            "min_loss": float(min(losses)),
            "max_loss": float(max(losses)),
        }
    all_losses = [r["loss"] for r in records]
    if all_losses:
        summary["overall"] = {
            "batches": len(all_losses),
            "mean_loss": float(sum(all_losses) / len(all_losses)),
            "min_loss": float(min(all_losses)),
            "max_loss": float(max(all_losses)),
        }
    return summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    include_tasks = None
    if args.include_tasks:
        include_tasks = [t.strip() for t in args.include_tasks.split(",") if t.strip()]
        for t in include_tasks:
            if t not in TASKS:
                raise ValueError(f"Unknown task '{t}'. Allowed: {', '.join(TASKS)}")

    task_samples = parse_task_samples(args.task_samples, include_tasks or list(TASKS))

    if include_tasks:
        dataset_tasks = include_tasks
    else:
        dataset_tasks = list(task_samples.keys())

    model, tokenizer = load_model(args)
    dataset, loader = build_eval_loader(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        task_samples={k: task_samples[k] for k in dataset_tasks},
        max_input=args.max_input_length,
        max_target=args.max_target_length,
        num_workers=args.num_workers,
    )

    print("Loaded dataset with counts:")
    for t in dataset.include_tasks:
        requested = task_samples.get(t)
        actual = len([ex for ex in dataset.examples if ex['task'] == t])
        print(f"  {t}: {requested} requested, {actual} actual")

    records = evaluate(model, loader, args.device, args.quiet)

    fig_path = os.path.join(args.output_dir, "loss_curve.png")
    plot_losses(records, fig_path)

    summary = summarize(records)
    report_path = os.path.join(args.output_dir, "loss_stats.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "ckpt_path": args.ckpt_path,
                "pretrained_model": args.pretrained_model,
                "batch_size": args.batch_size,
                "task_samples": task_samples,
            },
            "summary": summary,
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved stats to {report_path}")


if __name__ == "__main__":
    main()
